import argparse
import os
import sys
from importlib import import_module
from abp.configs import NetworkConfig, ReinforceConfig, EvaluationConfig

import torch
import numpy as np 
import torch.nn as nn

from abp.adaptives import TransAdaptive
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam

a_b_m = 0
a_b_v = 1
a_b_c = 2
a_b_p = 3
a_nex = 4
e_b_m = 5
e_b_v = 6
e_b_c = 7
e_b_p = 8
e_nex = 9
a_mnrl = 10
a_u_m = 11
a_u_v = 12
a_u_c = 13
e_u_m = 14
e_u_v = 15
e_u_c = 16
a_rwd = 17 # This entry only in second column of data ( so in entry data[i][1], NOT data[i][0] )

#####################################

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

#########################


def pre_process():
    data_1 = torch.load('100000_random_v_random.pt')

    data_2 = torch.load('60000_sadq_v_random.pt')
    print(len(data_1[0][0]))
    print(len(data_2[0][0]))
    print(len(data_1[0][1]))
    print(len(data_2[0][1]))

    data_1_games = grab_games(data_1)
    data_2_games = grab_games(data_2)

    data = data_1 + data_2

    return data


def grab_games(data):
    data_games = []
    data_games.append(0)
    for i in range(len(data)):
        if (((data[i][0][a_b_m]) + (data[i][0][a_b_v]) + (data[i][0][a_b_c]) + (data[i][0][a_b_p]) + 
            (data[i][0][e_b_m]) + (data[i][0][e_b_v]) + (data[i][0][e_b_c]) + (data[i][0][e_b_p])) < 
            ((data[i-1][0][a_b_m]) + (data[i-1][0][a_b_v]) + (data[i-1][0][a_b_c]) + (data[i-1][0][a_b_p]) + 
            (data[i-1][0][e_b_m]) + (data[i-1][0][e_b_v]) + (data[i-1][0][e_b_c]) + (data[i-1][0][e_b_p]))):
            data_games.append(i)
    
    return data_games

def preProcessData(data):

    for i in range(0, len(data)):
        data[i][1] = [data[i][1][27], data[i][1][28], data[i][1][29], data[i][1][30]]
    
    # print(data[0][0], data[0][1])

    return data

def normalize(np_data, output_shape):
    norm_vector_input = np.array([700, 50, 40, 20, 50, 40, 20, 3,
                                    50, 40, 20, 50, 40, 20, 3,
                                    50, 40, 20, 50, 40, 20, 
                                    50, 40, 20, 50, 40, 20,
                                    2000, 2000, 2000, 2000, 40])

    norm_vector_output = np.array([2000,2000,2000,2000]) 

    if len(np_data[0]) == output_shape:
        return np_data / norm_vector_output
    else:
        return np_data / norm_vector_input

def calculateBaseline(data, val_indices):
    test_data = [data[i] for i in val_indices]
    val_set = np.array(test_data)

    baseline = np.stack(val_set[:, 0])
    idx = [4, 9]
    baseline_hp = baseline[:, idx]

    bl_next_state_reward = np.stack(val_set[:, 1])

    mse_baseline = ((baseline_hp - bl_next_state_reward)**2).mean(axis=None)
    print(mse_baseline)
    return mse_baseline

def split_data(dataset, val_pct):
    # Determine size of validation set
    n_val = int(val_pct*dataset)
    # Create random permutation of 0 to n-1
    idxs = np.random.permutation(dataset)
    # Pick first n_val indices for validation set
    return idxs[n_val:], idxs[:n_val]

def run_task(evaluation_config, network_config, reinforce_config):

    trans_model = TransAdaptive(name= "TugOfWar2lNexusHealth",
                                network_config=network_config,
                                reinforce_config = reinforce_config)

    # data = pre_process()
    np.set_printoptions(suppress=True)

    data = torch.load('test_random_vs_random_2l.pt')

    np_data = np.array(data)
    data_input = np.stack(np_data[:,0])
    data_output = np.stack(np_data[:,1])

    nexus_idx = [27,28,29,30]
    data_output = data_output[:,nexus_idx]

    data_x = normalize(data_input, network_config.output_shape)
    data_y = normalize(data_output, network_config.output_shape)

    tensor_x = torch.stack([torch.Tensor(i) for i in data_x])
    tensor_y = torch.stack([torch.Tensor(i) for i in data_y])

    tensor_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)

    train_indices, val_indices = split_data(len(data), val_pct=0.1)

    # mse_baseline = calculateBaseline(data, val_indices)

    print(len(train_indices), len(val_indices))
    print(val_indices[:10]) 

    train_sampler = SubsetRandomSampler(train_indices)
    train_dl = DataLoader(tensor_dataset, reinforce_config.batch_size, sampler=train_sampler)

    val_sampler = SubsetRandomSampler(val_indices)
    valid_dl = DataLoader(tensor_dataset, reinforce_config.batch_size, sampler=val_sampler)

    device = get_default_device()

    train_dl = DeviceDataLoader(train_dl, device)
    valid_dl = DeviceDataLoader(valid_dl, device) 

    to_device(trans_model.trans_model.model, device)

    # trans_model = nn.DataParallel(trans_model, device_ids=[0,1,2])

    optimizer = Adam(trans_model.trans_model.model.parameters(), lr = 0.0001)
    loss_fn = nn.MSELoss(reduction='mean')

    losses, metrics = trans_model.train(10000, 0.0001, loss_fn, train_dl, valid_dl, accuracy, optimizer, "nexus-HP-transition-model-report/test_loss_2l.txt")

def accuracy(outputs, ground_true):
    preds = torch.sum( torch.eq( outputs, ground_true), dim=1 )
    return torch.sum(preds).item() / len(preds)

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-f', '--folder',
        help='The folder containing the config files',
        required=True
    )

    parser.add_argument(
        '--eval',
        help="Run only evaluation task",
        dest='eval',
        action="store_true"
    )

    parser.add_argument(
        '-r', '--render',
        help="Render task",
        dest='render',
        action="store_true"
    )

    args = parser.parse_args()

    evaluation_config_path = os.path.join(args.folder, "evaluation.yml")
    evaluation_config = EvaluationConfig.load_from_yaml(evaluation_config_path)

    network_config_path = os.path.join(args.folder, "network.yml")
    network_config = NetworkConfig.load_from_yaml(network_config_path)

    reinforce_config_path = os.path.join(args.folder, "reinforce.yml")
    reinforce_config = ReinforceConfig.load_from_yaml(reinforce_config_path)

    if args.eval:
        evaluation_config.training_episodes = 0
        network_config.restore_network = True

    if args.render:
        evaluation_config.render = True

    run_task(evaluation_config, network_config, reinforce_config)

    return 0


if __name__ == "__main__":
    main()