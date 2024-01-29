import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Optional, Union
import json
import logging
class HiddenStatesDataset(Dataset):
    '''
    Loads the hidden states dataset and text generations from files generated by `generate.py`.
    '''
    def __init__(self,
                 hs_file='hidden_states.pt',
                 tg_file='text_generations.csv',
                 precision=torch.float16,
                 layer_idx: Optional[Union[int, List[int]]] = None,  
                 rebalance: bool = True,
                 # layer_idx could be a list of integers or a single integer
                 device='cuda'):
        self.layer_idx = layer_idx
        hs = torch.load(hs_file, map_location=device).type(precision)
        assert hs.dim() in (2, 3)

        if self.layer_idx is None:
            logging.info('Using all layers')
            if hs.dim() == 3:
                hs = hs.reshape(hs.shape[0], -1)
        else:
            if isinstance(self.layer_idx, int):
                self.layer_idx = [self.layer_idx]
            assert all(isinstance(i, int) for i in self.layer_idx), "layer_idx must be a list of integers"
            logging.info('Using layers {}'.format(self.layer_idx))
            hs = hs[:, self.layer_idx, :].reshape(hs.shape[0], -1)

        self.hidden_states = hs
        logging.info('hidden_states.shape={}'.format(self.hidden_states.shape))
        self.text_generations: list = json.load(open(tg_file, 'r'))

        self.pik_labels = np.array([sample['evaluation'] for sample in self.text_generations])
        # unit_labels 
        self.unit_labels = []
        for sample in self.text_generations:
            self.unit_labels.extend(sample['unit_evaluations'])
        self.unit_labels = np.array(self.unit_labels).astype(int)    
        
        
        # ensure the len of unit_labels would be the multiple of len of hidden_states
        logging.info(f'len of unit_labels: {len(self.unit_labels)}')
        logging.info(f'len of hidden_states: {len(self.hidden_states)}')
        assert len(self.unit_labels) % len(self.hidden_states) == 0, 'len of unit_labels is not the multiple of len of hidden_states'
        
        self.per_unit_labels = len(self.unit_labels) // len(self.hidden_states)
        
        if rebalance:
            logging.info('Rebalancing dataset')
            logging.info('Before rebalancing, hidden_states.shape={}'.format(self.hidden_states.shape))
            self._rebalance()
            logging.info('After rebalancing, hidden_states.shape={}'.format(self.hidden_states.shape))
        
    def __len__(self):
        return len(self.unit_labels)

    def __getitem__(self, i):
        hid = i // self.per_unit_labels
        return (
            self.hidden_states[hid],
            self.unit_labels[i],
        )
  
    def get_pik_label(self, hid):
        return self.pik_labels[hid]
    
    
    def _rebalance(self, oversample=True):
        if not oversample:
            raise NotImplementedError("Undersampling not implemented")

        # Divide pik_labels into bins
        bins = np.linspace(0, 1, 11) # 10 bins from 0 to 1
        bin_indices = np.digitize(self.pik_labels, bins) - 1 # numpy's digitize returns 1-based indices

        # Count samples in each bin
        bin_counts = np.bincount(bin_indices, minlength=len(bins) - 1)
        logging.info("Before rebalancing, bin_counts={}".format(bin_counts))
        
        # Calculate maximum number of samples in any bin
        max_samples = np.max(bin_counts)

        # desired number of samples in each bin
        desired_samples = max_samples // 3
        
        # Identify underrepresented bins and calculate oversampling factor
        oversampling_factor = desired_samples / bin_counts
        
        # for factor < 1, set to 1
        oversampling_factor[oversampling_factor < 1] = 1
        
        # Duplicate samples
        new_hidden_states = []
        new_pik_labels = []
        for i in range(len(self.hidden_states)):
            bin_idx = bin_indices[i]
            factor = int(oversampling_factor[bin_idx])

            for _ in range(factor):
                new_hidden_states.append(self.hidden_states[i])
                new_pik_labels.append(self.pik_labels[i])
        
        # print the new bin counts
        new_bin_indices = np.digitize(new_pik_labels, bins) - 1
        new_bin_counts = np.bincount(new_bin_indices, minlength=len(bins) - 1)
        logging.info("After rebalancing, new_bin_counts={}".format(new_bin_counts))
        
        # Convert lists to appropriate formats
        self.hidden_states = torch.stack(new_hidden_states)
        self.pik_labels = np.array(new_pik_labels)

        # Shuffle dataset
        shuffling_idx = torch.randperm(len(self.hidden_states))
        self.hidden_states = self.hidden_states[shuffling_idx]
        self.pik_labels = self.pik_labels[shuffling_idx]