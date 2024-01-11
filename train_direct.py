import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_seed', type=int, default=101, help='seed for splitting hidden states into train and test')
    parser.add_argument('--train_seed', type=int, default=8421, help='seed for train reproducibility')
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
    return parser.parse_args()

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
            last_layer_only=False,
            device=args.device)
        
        self.model = LinearProbe(self.dataset.hidden_states.shape[-1]).to(args.device)
        
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
    # def validate_model(model, val_loader, args):
    #     loss_fn = torch.nn.BCELoss()
    #     val_losses = []
    #     model.eval()
    #     with torch.no_grad():
    #         for hs, labels in tqdm(val_loader, leave=False):
    #             hs, labels = hs.to(args.device), labels.to(args.device)
    #             outputs = model(hs)
    #             loss = loss_fn(outputs, labels)
    #             val_losses.append(loss.item())
    #     return val_losses
    

    # def evaluate_model(model, test_loader, args):
    #     loss_fn = torch.nn.BCELoss()
    #     test_losses = []
    #     model.eval()
    #     with torch.no_grad():
    #         for hs, labels in tqdm(test_loader, leave=False):
    #             hs, labels = hs.to(args.device), labels.to(args.device)
    #             outputs = model(hs)
    #             loss = loss_fn(outputs, labels)
    #             test_losses.append(loss.item())
    #     return test_losses

# def plot_losses(train_losses, test_losses):
#     plt.figure(figsize=(10, 5))
#     plt.plot(train_losses, label='Train Loss')
#     plt.plot(test_losses, label='Test Loss')
#     plt.title('Training and Testing Losses')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.show()

# def calculate_brier_scores(df):
#     df['sq_errors'] = (df['evaluation'] - df['prediction']) ** 2
#     train_brier = df[df['split'] == 'train']['sq_errors'].mean()
#     test_brier = df[df['split'] == 'test']['sq_errors'].mean()
#     return train_brier, test_brier

# def plot_calibration(calib):
#     sns.relplot(data=calib, x='evaluation', y='prediction', hue='split', aspect=1.0, height=6)
#     plt.title('Calibration Plot')
#     plt.xlabel('Evaluation')
#     plt.ylabel('Prediction')
#     plt.show()

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
    
    # plot the training loss
    # plt.figure(figsize=(10, 5))
    # plt.plot(training_loss)
    # plt.title('Training Loss')
    # plt.xlabel('Steps')
    # plt.ylabel('Loss')
    

    # train_loader, val_loader, test_loader, dataset = load_and_split_dataset(args)

    # model = LinearProbe(dataset.hidden_states.shape[-1]).to(args.device)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    # train_losses = train_model(model, train_loader, args)
    # test_losses = evaluate_model(model, test_loader, args)

    # plot_losses(train_losses, test_losses)

    # # Additional logic for predictions, calibration, and Brier scores
    # train_brier, test_brier = calculate_brier_scores(df)
    # print(f'Train Brier score: {train_brier:.4f}, Test Brier score: {test_brier:.4f}')

    # plot_calibration(calib)
