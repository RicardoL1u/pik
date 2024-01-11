import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class DirectHiddenStatesDataset(Dataset):
    '''
    Loads the hidden states dataset and text generations from files generated by `generate.py`.
    '''
    def __init__(self,
        hs_file='hidden_states.pt',
        tg_file='text_generations.csv',
        precision=torch.float16,
        last_layer_only=False,
        device='cuda',
    ):
        hs = torch.load(hs_file, map_location=device).type(precision)
        assert hs.dim() in (2, 3)
        if not last_layer_only and hs.dim() == 3:
            # HACK: to keep the fisrt half layers
            # hs = hs[:, :hs.shape[1] // 2, :]
            # HACK: to keep the middle half layers
            # hs = hs[:, hs.shape[1] // 4:hs.shape[1] // 4 * 3, :]
            hs = hs.reshape(hs.shape[0], hs.shape[1] * hs.shape[2])
        elif last_layer_only and hs.dim() == 3:
            hs = hs[:, -1, :]
            # HACK: to use the second last layer, we take the last layer of the reversed tensor
            # hs = hs[:, -2, :]
        self.hidden_states = hs
        self.text_generations = pd.read_csv(tg_file)
        
        # Compute mean evaluations for each 'hid' in a more efficient way
        mean_evaluations = self.text_generations.groupby('hid')['evaluation'].mean().to_dict()

        # Create pik_labels using a vectorized operation
        self.pik_labels = np.array([mean_evaluations.get(hid, np.nan) for hid in range(self.hidden_states.shape[0])])

    def __len__(self):
        return self.hidden_states.shape[0]

    def __getitem__(self, i):
        return (
            self.hidden_states[i],
            self.pik_labels[i]
        )
  
    def get_pik_label(self, hid):
        return self.pik_labels[hid]