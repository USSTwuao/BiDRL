from torch.utils.data import Dataset
import torch
import os
import pickle
from problems.state_mfvrp import StateMFVRP
from generate_MFVRP_data import generate_MFVRP_data


class MFVRP(object):
    NAME = 'mfvrp'


    @staticmethod
    def make_state(*args, **kwargs):
        return StateMFVRP.initialize(*args, **kwargs)
    

    @staticmethod
    def get_costs(state):
        cost1 = state.lengths 
        cost2 = state.cur_Ecost 
        cost = torch.cat((cost1*0.5,cost2),dim = 1)
        return cost


    @staticmethod
    def make_dataset(*args, **kwargs):
        return MFVRPDataset(*args, **kwargs)

def make_instance(args): 
    loc, demand, timewindow ,servetime, *args = args  

    grid_size = 1
    return {
        'loc': torch.tensor(loc, dtype=torch.float) / grid_size,
        'demand': torch.tensor(demand, dtype=torch.float),
        'timewindow': torch.tensor(timewindow, dtype=torch.float) / grid_size,
        'servetime': torch.tensor(servetime, dtype=torch.float)
    }


class MFVRPDataset(Dataset):
    def __init__(self, size, num_samples, filename=None, offset=0, distribution=None):
        super(MFVRPDataset, self).__init__()

        

        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)

            self.data = [make_instance(args) for args in data[offset:offset + num_samples]]

        else:
            self.data = generate_MFVRP_data(num_samples,size)
            self.size = len(self.data) 

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx] 
