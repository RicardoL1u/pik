import argparse
import pandas as pd
import numpy as np
from transformers import get_scheduler
import torch
from torch.utils.data import DataLoader, Subset
from pik.datasets.direct_hidden_states_dataset import DirectHiddenStatesDataset
from pik.datasets.hidden_states_dataset import HiddenStatesDataset
from pik.models.probe_model import LinearProbe, MLPProbe
from tqdm import tqdm
import os
import wandb
import time
from pik.utils.utils import wandb_log
from pik.utils.try_to_plot import plot_calibration, plot_and_save_scatter, plot_training_loss
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Function Definitions


# set seed
import random
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

def parse_arguments():
    def parse_layers(arg):
        if arg.lower() == 'all':
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
    parser.add_argument('--direct', action='store_true', default=False, help='whether to use direct hidden states')
    parser.add_argument('--rebalance', action='store_true', default=False, help='whether to rebalance the dataset')
    parser.add_argument('--debug', action='store_true', default=False, help='set to True to enable debug mode')
    parser.add_argument('--model_layer_idx', default=None, type=parse_layers,
                    help='Model layer index(es), which layer(s) to use. None for all layers, \
                    or specify indices separated by commas (e.g., 0,2,4).')
    args = parser.parse_args()
    # Set logging level
    logging.getLogger().setLevel(args.logging_level)
    logging.info(f'Logging level set to {args.logging_level}')
    
    fh = logging.FileHandler(os.path.join(args.output_dir, f'train_direct_{time.strftime("%Y%m%d-%H%M%S")}.log'))
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
        # torch.set_default_dtype(args.precision)
        
        logging.info("===== Loading Data =====")
        dataset_cls = DirectHiddenStatesDataset if args.direct else HiddenStatesDataset
        logging.info("Using {} dataset".format(dataset_cls.__name__))
        
        self.loss_fn = torch.nn.MSELoss() if args.direct else torch.nn.BCEWithLogitsLoss()
        logging.info("Using {} loss function".format(self.loss_fn.__class__.__name__))
        
        # Load dataset...
        self.dataset = dataset_cls(
            hs_file=args.hidden_states_filename,
            tg_file=args.text_generations_filename,
            precision=args.precision,
            layer_idx=args.model_layer_idx,
            # rebalance=args.rebalance,
            device=args.device
        )
        
        self.model = self.probe_cls(self.dataset.hidden_states.shape[-1]).to(args.device)
        # set precision
        if args.precision == torch.float16:
            self.model = self.model.half()
        # use xavier initialization for all the layers
        # for name, param in self.model.named_parameters():
        #     if 'weight' in name:
        #         torch.nn.init.xavier_uniform_(param)
        
        
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
        
        if args.debug:
            self.train_hids = self.train_hids[:100]
            self.val_hids = self.val_hids[:100]
            self.test_hids = self.test_hids[:100]
        
        if args.rebalance:
            
            bins = np.linspace(0, 1, 11)
            train_labels = self.dataset.pik_labels[self.train_hids]
            bin_indices = np.digitize(train_labels, bins) - 1 # numpy's digitize returns 1-based indices
            
            # Count samples in each bin
            bin_counts = np.bincount(bin_indices, minlength=len(bins) - 1)
            logging.info("Before rebalancing, bin_counts={}, toal hid={}".format(bin_counts, len(self.train_hids)))
            
            # Calculate maximum number of samples in any bin
            max_samples = np.max(bin_counts)

            # desired number of samples in each bin
            desired_samples = 1000
            
            # Identify underrepresented bins and calculate oversampling factor
            oversampling_factor = desired_samples / bin_counts
            
            # for factor < 1, set to 1
            # oversampling_factor[oversampling_factor < 1] = 1
            
            # repeat the hids based on the oversampling factor
            new_train_hids = []
            for i in range(len(self.train_hids)):
                bin_idx = bin_indices[i]
                if oversampling_factor[bin_idx] > 1:
                    factor = int(oversampling_factor[bin_idx])
                    new_train_hids.extend([self.train_hids[i]] * factor)
                else:
                    if random.random() < oversampling_factor[bin_idx]:
                        new_train_hids.append(self.train_hids[i])
            
            new_train_labels = self.dataset.pik_labels[new_train_hids]
            new_bin_indices = np.digitize(new_train_labels, bins) - 1
            new_bin_counts = np.bincount(new_bin_indices, minlength=len(bins) - 1)
            logging.info("After rebalancing, new_bin_counts={}, toal hid={}".format(new_bin_counts, len(new_train_hids)))
            
            self.train_hids = new_train_hids
         
        if not args.direct:
            # need convert the hid to index
            self.train_idx = []
            
            # for example the per_unit_labels is 5, hids = [0,2,7]
            # idx = [0,1,2,3,4,10,11,12,13,14,35,36,37,38,39]
            wandb_log(logging.INFO, self.use_wandb, "There are {} per unit labels".format(self.dataset.per_unit_labels))
            wandb_log(logging.INFO, self.use_wandb, "Converting hids to idx...")
            for hid in self.train_hids:
                self.train_idx.extend([hid * self.dataset.per_unit_labels + i for i in range(self.dataset.per_unit_labels)])
        
        else:
            self.train_idx = self.train_hids 
        
        wandb_log(logging.INFO, self.use_wandb, "There are {} train hids".format(len(self.train_hids)))
        wandb_log(logging.INFO, self.use_wandb, "There are {} val hids".format(len(self.val_hids)))
        wandb_log(logging.INFO, self.use_wandb, "There are {} test hids".format(len(self.test_hids)))


        # Create DataLoader for train and test sets...
        self.train_loader = DataLoader(Subset(self.dataset, self.train_idx), batch_size=args.batch_size, shuffle=True)
        logging.info("There are {} samples and {} batches in the train loader".\
                        format(len(self.train_loader.dataset), len(self.train_loader)))
        
    def trainning_loop(self):
        
        self.model.to(self.args.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        # use l2 loss
        loss_fn = self.loss_fn.to(self.args.device)
        
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
                loss += 0.001 * l2_reg
                
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
        if self.args.debug:
            # for train set
            val_brier = train_brier
        return train_brier, val_brier, test_brier
    
    def plot_scatters_to_wandb(self, preds, labels, train_losses):
        # plot scatter
        df = pd.DataFrame({'evaluation': labels, 'prediction': preds})
        df.loc[self.test_hids, 'split'] = 'test'
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
        
        plot_calibration(test_evals, test_preds, 20,
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
    
