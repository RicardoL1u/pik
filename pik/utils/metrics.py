import torch
import numpy as np
from sklearn.metrics import roc_auc_score

def calculate_brier_score(predictions:torch.Tensor, labels:torch.Tensor) -> float:
    '''
    Calculates the Brier score for a set of predictions and labels.
    '''
    return torch.mean((predictions - labels)**2).item()

def calculate_ECE_quantile(predictions: torch.Tensor, labels: torch.Tensor, bins: int = 10) -> float:
    '''
    Calculates the Expected Calibration Error for a set of predictions and labels
    in a regression context with quantile binning, where each bin has an equal number of predictions.
    '''
    # Flatten the tensors and convert to numpy for easier manipulation
    confidences = predictions.flatten().cpu().numpy()
    labels = labels.flatten().cpu().numpy()
    
    # Sort confidences and labels together based on confidences
    sorted_indices = np.argsort(confidences)
    sorted_confidences = confidences[sorted_indices]
    sorted_labels = labels[sorted_indices]
    
    # Calculate the number of predictions per bin
    num_predictions = len(confidences)
    predictions_per_bin = num_predictions // bins
    
    ece = 0.0
    for i in range(bins):
        # Determine the start and end indices for this bin
        start_index = i * predictions_per_bin
        # For the last bin, include any leftover predictions
        if i == bins - 1:
            end_index = num_predictions
        else:
            end_index = (i + 1) * predictions_per_bin
        
        # Extract the confidences and labels for this bin
        bin_confidences = sorted_confidences[start_index:end_index]
        bin_labels = sorted_labels[start_index:end_index]
        
        # Calculate the average confidence and observed frequency in the bin
        avg_confidence_in_bin = np.mean(bin_confidences)
        observed_freq_in_bin = np.mean(bin_labels)
        
        # Calculate the absolute difference between average confidence and observed frequency
        bin_error = np.abs(avg_confidence_in_bin - observed_freq_in_bin)
        
        # Update the ECE
        ece += bin_error * (end_index - start_index) / num_predictions
    
    return ece



# test utils
if __name__ == '__main__':
    # Generating synthetic data for testing
    np.random.seed(42)  # For reproducibility
    torch.manual_seed(42)

    # Synthetic regression data
    num_samples = 1000
    predictions = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    labels = torch.tensor([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
    
    # Calculate Brier score
    brier_score = calculate_brier_score(predictions, labels)
    assert np.isclose(brier_score, 0.05**2, atol=1e-3), f'Expected brier_score to be 0.05^2, but got {brier_score}'
    print('Brier score:', brier_score)
    
    # Calculate ECE
    ece = calculate_ECE_quantile(predictions, labels, bins=10)
    assert np.isclose(ece, 0.05, atol=1e-3), f'Expected ece to be 0.05, but got {ece}'
    print('ECE:', ece)