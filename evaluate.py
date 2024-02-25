from pik.datasets.direct_hidden_states_dataset import DirectHiddenStatesDataset
from pik.models.probe_model import LinearProbe, MLPProbe
import argparse
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from pik.utils.try_to_plot import plot_calibration
from pik.utils.metrics import calculate_brier_score, calculate_ECE_quantile

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
def parse_arguments():
    def parse_layers(arg):
        if arg.lower() == 'none':
            return None
        try:
            return [int(layer.strip()) for layer in arg.split(',')]
        except ValueError:
            raise argparse.ArgumentTypeError("Value must be 'None' or a comma-separated list of integers")
    parser = argparse.ArgumentParser(description='Evaluate a linear probe on a dataset.')
    parser.add_argument('--dataset', required=True, choices=['gsm8k','trivia_qa','commonsense_qa'], help='dataset to use')
    parser.add_argument('--model_ckpt_path', required=True, help='path to model checkpoint')
    parser.add_argument('--hidden_states_filename', default='hidden_states.pt', help='filename for saving hidden states')
    parser.add_argument('--text_generations_filename', default='text_generations.csv', help='filename for saving text generations')
    parser.add_argument('--debug', action='store_true', help='Debug mode.')
    parser.add_argument('--device', default='cuda', help='Device to use.')
    parser.add_argument('--model_layer_idx', default=None, type=parse_layers,
                    help='Model layer index(es), which layer(s) to use. None for all layers, \
                    or specify indices separated by commas (e.g., 0,2,4).')
    parser.add_argument('--mlp', action='store_true', help='Use MLP probe instead of linear probe.')
    args = parser.parse_args()
    return args

def setup_logging(args):
    logger = logging.getLogger(__name__)
    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    # logging all the args
    logging.info("===== Args =====")
    for arg in vars(args):
        logging.info("{}: {}".format(arg, getattr(args, arg)))
    return logger

if __name__ == '__main__':
    
    
    args = parse_arguments()
    if 'cuda' in args.device:
        torch.set_default_dtype(torch.float16)
        precision = torch.float16
    else:
        torch.set_default_dtype(torch.float32)
        precision = torch.float32
        
    logger = setup_logging(args)
    logging.info("Loading dataset from {} and {}" \
                 .format(args.hidden_states_filename, args.text_generations_filename))
    dataset = DirectHiddenStatesDataset(
        hs_file = args.hidden_states_filename, 
        tg_file = args.text_generations_filename,
        layer_idx = args.model_layer_idx,
        precision=precision,
        device = args.device,
        rebalance=False
    )
    
    probe_cls = MLPProbe if args.mlp else LinearProbe
    logging.info("Use {} probe".format('MLP' if args.mlp else 'linear'))
    model = probe_cls(dims=dataset.hidden_states.shape[1]).to(args.device)
    logging.info("Loading model checkpoint from {}".format(args.model_ckpt_path))
    model.load_state_dict(torch.load(args.model_ckpt_path))
    # evaluate
    model.eval()
    with torch.no_grad():
        preds: torch.Tensor = model(dataset.hidden_states).detach().cpu().squeeze()
    labels:torch.Tensor = dataset.pik_labels
    
    logging.info("Mean prediction: {}, Mean label: {}".format(preds.mean(), labels.mean()))
    
    # calculate brier score
    brier_score = calculate_brier_score(preds, labels)
    ece = calculate_ECE_quantile(preds, labels, bins=10)
    logging.info("Brier score: {}, ECE: {}".format(brier_score, ece))
    
    # plt the scatter plot
    figure_path = args.model_ckpt_path.replace('.pt', args.dataset+'_scatter.png')
    plt.figure(figsize=(8,8))
    plt.scatter(preds, labels, alpha=0.5)
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.savefig(figure_path)
    
    plot_calibration(preds, labels, num_bins=10, file_name=figure_path.replace('.png', '_calibration.png'))