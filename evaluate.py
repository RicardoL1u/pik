from pik.datasets.direct_hidden_states_dataset import DirectHiddenStatesDataset
from pik.models.linear_probe import LinearProbe
import argparse
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from try_to_plot import plot_calibration

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
    parser.add_argument('--dataset', required=True, choices=['gsm8k','trivia_qa'], help='dataset name')
    parser.add_argument('--model_ckpt_path', required=True, help='path to model checkpoint')
    parser.add_argument('--hidden_states_filename', default='hidden_states.pt', help='filename for saving hidden states')
    parser.add_argument('--text_generations_filename', default='text_generations.csv', help='filename for saving text generations')
    parser.add_argument('--debug', action='store_true', help='Debug mode.')
    parser.add_argument('--model_layer_idx', default=None, type=parse_layers,
                    help='Model layer index(es), which layer(s) to use. None for all layers, \
                    or specify indices separated by commas (e.g., 0,2,4).')
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
    
    torch.set_default_dtype(torch.float16)
    
    args = parse_arguments()
    logger = setup_logging(args)
    logging.info("Loading dataset from {} and {}" \
                 .format(args.hidden_states_filename, args.text_generations_filename))
    dataset = DirectHiddenStatesDataset(
        hs_file = args.hidden_states_filename, 
        tg_file = args.text_generations_filename,
        layer_idx = args.model_layer_idx
    )
    model = LinearProbe(dims=dataset.hidden_states.shape[1]).to('cuda')
    logging.info("Loading model checkpoint from {}".format(args.model_ckpt_path))
    try:
        model.load_state_dict(torch.load(args.model_ckpt_path))
    except:
        logging.info("Loading model checkpoint from {} failed, \
                     trying to convert the keys".format(args.model_ckpt_path))
        # convert the keys to from ln.weight to model.0.weight and ln.bias to model.0.bias
        ckpt_dict = torch.load(args.model_ckpt_path)
        new_ckpt_dict = {}
        for k, v in ckpt_dict.items():
            new_ckpt_dict['model.0.'+k.replace('ln.','')] = v
        model.load_state_dict(new_ckpt_dict)
    # evaluate
    model.eval()
    with torch.no_grad():
        preds = model(dataset.hidden_states).detach().cpu().numpy().squeeze()
    labels = dataset.pik_labels
    
    # calculate brier score
    brier_score = np.mean((preds - labels)**2)
    logging.info("Brier score: {}".format(brier_score))    
    
    # plt the scatter plot
    figure_path = args.model_ckpt_path.replace('.pt', args.dataset+'_scatter.png')
    plt.figure(figsize=(8,8))
    plt.scatter(labels, preds, alpha=0.5)
    plt.xlabel('pik labels')
    plt.ylabel('predictions')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.savefig(figure_path)
    
    plot_calibration(labels, preds, num_bins=10, file_name=figure_path.replace('.png', '_calibration.png'))