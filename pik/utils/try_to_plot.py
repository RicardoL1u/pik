import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List
def plot_calibration(x_list, y_list, num_bins=10, file_name='calibration_plot.png'):
    # Zip and sort the x and y values
    xy_tuples = sorted(zip(x_list, y_list), key=lambda x: x[0])

    # bins = np.linspace(0, 1, num_bins + 1)
    
    # # group xy_tuples into bins, each bin is a list of tuples
    # xy_grouped = [[sample for sample in xy_tuples if sample[0] >= bins[i] and sample[0] < bins[i+1]]
    #               for i in range(num_bins)]
    
    # Group into bins
    total_num = len(xy_tuples)
    xy_grouped = [xy_tuples[int(i * total_num / num_bins): int((i + 1) * total_num / num_bins)] 
                  for i in range(num_bins)]

    # Calculate mean for each bin
    x_means = [sum([x[0] for x in group]) / len(group) for group in xy_grouped]
    y_means = [sum([x[1] for x in group]) / len(group) for group in xy_grouped]

    # Plotting
    plt.figure(figsize=(10, 10))
    plt.scatter(x_means, y_means, s=100, c='red', marker='o', label='bins')
    plt.plot(x_means, y_means, c='red', label='line')
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.title('Calibration Plot')
    plt.legend()
    
    # Save the plot
    plt.savefig(file_name)

def plot_training_loss(loss_list, logging_steps, file_name):
    # Plotting
    plt.figure(figsize=(10, 10))
    plt.plot([i * logging_steps for i in range(len(loss_list))], loss_list, c='red', label='line')
    plt.xlabel('Training Steps')
    plt.ylabel('Training Loss')
    plt.title('Training Loss')
    plt.legend()
    
    # Save the plot
    plt.savefig(file_name)

def plot_metrics(metrics:List[dict], epochs:int, metric_name:str, file_name:str):
    # Plotting
    plt.figure(figsize=(10, 10))
    for split, color in zip(['train', 'val', 'test'], ['red', 'blue', 'green']):
        split_metrics = [metric[f'{split}_{metric_name}'] for metric in metrics]
        plt.plot([i for i in range(epochs)], split_metrics, c=color, label=split)
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name.capitalize()}')
    plt.legend()
    
    # Save the plot
    plt.savefig(file_name)
    
def plot_and_save_scatter(df, output_dir):
    """
    Plots and saves scatter plots of predictions vs consistencys for each split and combined.

    Args:
    df (DataFrame): The DataFrame containing the data.
    output_dir (str): The directory where the plots will be saved.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Function to plot individual scatter plot for each split
    def plot_split(df_split, color, label, file_suffix):
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.scatter(df_split['consistency'], df_split['prediction'], s=2, c=color, marker='o', label=label)
        plt.plot([0, 1], [0, 1], color='black', linestyle='--')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel('consistency')
        plt.ylabel('Prediction')
        plt.title(f'Scatter Plot - {label.capitalize()}')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'scatter_{file_suffix}.png'))

    # Plot for each split
    for split in ['train', 'val', 'test']:
        split_df = df[df['split'] == split]
        color = {'train': 'red', 'val': 'blue', 'test': 'green'}[split]
        plot_split(split_df, color, split, split)

    # Combined scatter plot
    fig, ax = plt.subplots(figsize=(10, 10))
    for split, color in zip(['train', 'val', 'test'], ['red', 'blue', 'green']):
        split_df = df[df['split'] == split]
        plt.scatter(split_df['prediction'], split_df['consistency'], s=100, c=color, marker='o', label=split)
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.title('Combined Scatter Plot')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'scatter_combined.png'))

# Example usage
if __name__ == '__main__':
    df = pd.read_csv('scatter.csv')
    df_test = df[df['split'] == 'test']
    eval_test = df_test['consistency'].tolist()
    pred_test = df_test['prediction'].tolist()

    plot_calibration(eval_test, pred_test, 10)
