import argparse
import pandas as pd
import numpy as np
from transformers import get_scheduler
import torch
from torch.utils.data import DataLoader, Subset
from pik.datasets.direct_hidden_states_dataset import DirectHiddenStatesDataset
from pik.models.linear_probe import LinearProbe
from tqdm import tqdm
import os
import wandb
from pik.utils import wandb_log
from try_to_plot import plot_calibration, plot_and_save_scatter, plot_training_loss
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Function Definitions

def parse_arguments():
    def parse_layers(arg):
        if arg.lower() == 'none':
            return None
        try:
            return [int(layer.strip()) for layer in arg.split(',')]
        except ValueError:
            raise argparse.ArgumentTypeError("Value must be 'None' or a comma-separated list of integers")
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_frac', type=float, default=0.8, help='fraction of hidden states to use for training')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=25, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--precision', default='float16', help='model precision')
    parser.add_argument('--device', default='cuda:0', help='device to run on')
    parser.add_argument('--use_wandb', action='store_true', default=False, help='set to True to use wandb')
    # parser.add_argument('--data_folder', default='data', help='data folder')
    parser.add_argument('--hidden_states_filename', default='hidden_states.pt', help='filename for saving hidden states')
    parser.add_argument('--text_generations_filename', default='text_generations.csv', help='filename for saving text generations')
    parser.add_argument('--output_dir', required=True, help='output directory')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='warmup ratio')
    parser.add_argument('--wandb_run_name', default='linear_probe', help='wandb run name')
    parser.add_argument('--logging_steps', type=int, default=10, help='logging steps')
    parser.add_argument('--logging_level', default='INFO', help='logging level')
    parser.add_argument('--model_layer_idx', default=None, type=parse_layers,
                    help='Model layer index(es), which layer(s) to use. None for all layers, \
                    or specify indices separated by commas (e.g., 0,2,4).')
    args = parser.parse_args()
    # Set logging level
    logging.getLogger().setLevel(args.logging_level)
    logging.info(f'Logging level set to {args.logging_level}')
    
    fh = logging.FileHandler(os.path.join(args.output_dir, 'train_direct.log'))
    fh.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    # Add the file handler to the root logger
    logging.getLogger('').addHandler(fh)
    
    
    # logging all the args
    logging.info("===== Args =====")
    for arg in vars(args):
        logging.info("{}: {}".format(arg, getattr(args, arg)))
    return args

class Trainer:
    def __init__(self, args):
        self.args = args
        self.use_wandb = args.use_wandb
        self.args.precision = torch.float16 if args.precision == 'float16' else torch.float32
        torch.set_default_dtype(args.precision)
        
        # Load dataset...
        self.dataset = DirectHiddenStatesDataset(
            hs_file=args.hidden_states_filename,
            tg_file=args.text_generations_filename,
            precision=args.precision,
            layer_idx=args.model_layer_idx,
            device=args.device)
        
        self.model = LinearProbe(self.dataset.hidden_states.shape[-1]).to(args.device)
        # use xavier initialization
        torch.nn.init.xavier_uniform_(self.model.ln.weight)
        
        
        wandb_log(logging.INFO, self.use_wandb,
                  f"The shape of hidden states is {self.dataset.hidden_states.shape}")
        
        # Split dataset into training and testing...
        permuted_hids = torch.randperm(self.dataset.hidden_states.shape[0]).tolist()
        train_len = int(args.train_frac * self.dataset.hidden_states.shape[0])
        
        # the rest hidden states are used for testing and validation
        val_len = int((1 - args.train_frac) * self.dataset.hidden_states.shape[0]) // 2
        test_len = int((1 - args.train_frac) * self.dataset.hidden_states.shape[0]) - val_len
        self.train_hids, \
        self.val_hids, \
        self.test_hids = \
            permuted_hids[:train_len], permuted_hids[train_len:train_len+val_len], permuted_hids[train_len+val_len:]
        
        wandb_log(logging.INFO, self.use_wandb, "There are {} train hids".format(len(self.train_hids)))
        wandb_log(logging.INFO, self.use_wandb, "There are {} val hids".format(len(self.val_hids)))
        wandb_log(logging.INFO, self.use_wandb, "There are {} test hids".format(len(self.test_hids)))

        
        # Create DataLoader for train and test sets...
        self.train_loader = DataLoader(Subset(self.dataset, self.train_hids), batch_size=args.batch_size, shuffle=True)
        self.val_loader = DataLoader(Subset(self.dataset, self.test_hids), batch_size=args.batch_size, shuffle=True)
        self.test_loader = DataLoader(Subset(self.dataset, self.test_hids), batch_size=args.batch_size, shuffle=True)
        
        
    def trainning_loop(self):
        
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate)
        # use l2 loss
        loss_fn = torch.nn.MSELoss()
        
        train_losses = []

        train_metrics = []
        val_metrics = []
        test_metrics = []
        
        wandb_log(logging.INFO, self.use_wandb, "===== Start Training =====")
        # log the number of epochs
        wandb_log(logging.INFO, self.use_wandb, "There are {} epochs".format(self.args.num_epochs))
        # log the number of steps
        # Total number of training steps
        total_steps = len(self.train_loader) * self.args.num_epochs
        wandb_log(logging.INFO, self.use_wandb, "There are {} steps".\
                  format(total_steps))
        
        # Initialize the learning rate scheduler
        scheduler = get_scheduler(
                                'constant' if self.args.warmup_ratio > 1 else 'linear',
                                optimizer=optimizer, 
                                num_warmup_steps=int(total_steps * self.args.warmup_ratio), 
                                num_training_steps=total_steps)
        
        step_now = 0
        running_loss = 0.0
        self.model.train()
        best_val_brier = 1e10
        for epoch in tqdm(range(self.args.num_epochs)):
            for hs, labels in self.train_loader:
                hs = hs.to(self.args.device)
                labels = labels.unsqueeze(1).type(self.args.precision).to(self.args.device)
                optimizer.zero_grad()
                outputs = self.model(hs)
                
                loss = loss_fn(outputs, labels)
                # Calculate L2 Regularization
                l2_reg = torch.tensor(0.).to(self.args.device)
                for param in self.model.parameters():
                    l2_reg += torch.norm(param)**2
                loss += 0.01 * l2_reg
                
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                step_now += 1
                scheduler.step()
                if step_now % self.args.logging_steps == 0:
                    wandb_log(logging.INFO, self.use_wandb, 
                              score_dict={"train_loss": running_loss / self.args.logging_steps,
                                          "learning_rate": scheduler.get_last_lr()[0],
                                          "step": step_now},
                              step=step_now)
                    train_losses.append(running_loss / self.args.logging_steps)
                    running_loss = 0.0
            # train_losses.append(running_loss / len(self.train_loader))
            
            # validate model
            all_preds, all_labels = self.prediction()
            train_brier, val_brier, test_brier = self.calculate_metrics(all_preds, all_labels)
            train_metrics.append(train_brier)
            val_metrics.append(val_brier)
            test_metrics.append(test_brier)
            wandb_log(logging.INFO, self.use_wandb, score_dict={"epoch": epoch,
                       "train_brier": train_brier, "val_brier": val_brier, "test_brier": test_brier},
                      step=step_now)
            # save the model with the best val brier
            if best_val_brier > val_brier:
                logging.info("The previous best val brier is {}, the current val brier is {}".\
                             format(best_val_brier, val_brier))
                logging.info("Saving the model...")
                best_val_brier = val_brier
                torch.save(self.model.state_dict(), os.path.join(self.args.output_dir, 'best_model.pt'))
            
        # load the best model
        logging.info("Loading the best model with val brier {}".format(best_val_brier))
        self.model.load_state_dict(torch.load(os.path.join(self.args.output_dir, 'best_model.pt')))
        # test the best model
        all_preds, all_labels = self.prediction()
        self.plot_scatters_to_wandb(all_preds, all_labels, train_losses)
        # Close wandb run
        if self.use_wandb:
            wandb.finish()   
        return train_losses, train_metrics, val_metrics, test_metrics

    def prediction(self):
        all_hs = self.dataset.hidden_states
        self.model.eval()
        with torch.inference_mode():
            all_preds = self.model(all_hs).detach().cpu().numpy().squeeze()
        all_labels = self.dataset.pik_labels
        return all_preds, all_labels
        
    def calculate_metrics(self, preds, labels):
        # calculate brier score
        # for train set
        train_brier = np.mean((labels[self.train_hids] - preds[self.train_hids]) ** 2)
        # for val set
        val_brier = np.mean((labels[self.val_hids] - preds[self.val_hids]) ** 2)
        # for test set
        test_brier = np.mean((labels[self.test_hids] - preds[self.test_hids]) ** 2)
        return train_brier, val_brier, test_brier
    
    def plot_scatters_to_wandb(self, preds, labels, train_losses):
        # plot scatter
        df = pd.DataFrame({'evaluation': labels, 'prediction': preds})
        df['split'] = 'test'
        df.loc[self.train_hids, 'split'] = 'train'
        df.loc[self.val_hids, 'split'] = 'val'

        train_brier, val_brier, test_brier = self.calculate_metrics(preds, labels)
        logging.info("The train brier is {}, the val brier is {}, the test brier is {}".\
                        format(train_brier, val_brier, test_brier))
        
        # save the dataframe to local
        df.to_csv(os.path.join(self.args.output_dir, 'scatters.csv'), index=False)
        if self.use_wandb:
            wandb.log({"train scatter": wandb.plot.scatter(wandb.Table(data=df[df['split'] == 'train']),
                                                    "evaluation", "prediction", title="Train Scatter")})
            wandb.log({"val scatter": wandb.plot.scatter(wandb.Table(data=df[df['split'] == 'val']),
                                                    "evaluation", "prediction", title="Val Scatter")})
            wandb.log({"test scatter": wandb.plot.scatter(wandb.Table(data=df[df['split'] == 'test']), 
                                                    "evaluation", "prediction", title="Test Scatter")})

        # get prediction and evaluation for each split
        train_preds = df[df['split'] == 'train']['prediction'].tolist()
        train_evals = df[df['split'] == 'train']['evaluation'].tolist()
        
        val_preds = df[df['split'] == 'val']['prediction'].tolist()
        val_evals = df[df['split'] == 'val']['evaluation'].tolist()
        
        test_preds = df[df['split'] == 'test']['prediction'].tolist()
        test_evals = df[df['split'] == 'test']['evaluation'].tolist()
        
        plot_calibration(test_evals, test_preds, 
                         file_name=os.path.join(self.args.output_dir, 'calibration.png'))
        plot_and_save_scatter(df, self.args.output_dir)
        plot_training_loss(train_losses, self.args.logging_steps, 
                           file_name=os.path.join(self.args.output_dir, 'training_loss.png'))

# Main Execution

if __name__ == "__main__":
    args = parse_arguments()
    
    # Ensure data files exist
    assert os.path.exists(args.hidden_states_filename)
    assert os.path.exists(args.text_generations_filename)
    
    if args.use_wandb:
        wandb.init(project="pik", name=args.wandb_run_name)
    
    trainer = Trainer(args)

    training_loss, train_metrics, val_metrics, test_metrics = trainer.trainning_loop()
    
