import torch
import numpy as np
import tqdm
from tensorboardX import SummaryWriter

from abp.utils import clear_summary_path
from abp.models.trans_model_tow import TransModel

import pickle
import os
import logging
import time
import random
from abp.adaptives.common.prioritized_memory.memory import ReplayBuffer
from copy import deepcopy

logger = logging.getLogger('root')
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
IntTensor = torch.cuda.IntTensor if use_cuda else torch.IntTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


class TransAdaptive(object):
    """ Adaptive that uses Transition Model """

    def __init__(self, name, network_config, reinforce_config):
        super(TransAdaptive, self).__init__()
        self.name = name
        self.network_config = network_config
        self.reinforce_config = reinforce_config
        
        self.memory = ReplayBuffer(self.reinforce_config.memory_size)
        # Global
        self.steps = 0
#         self.batch_size = 128

        reinforce_summary_path = self.reinforce_config.summaries_path + "/" + self.name

        if self.network_config.restore_network:
            restore_path = self.network_config.network_path + "/adaptive.info"
            self.memory.load(self.network_config.network_path)
            print("memory length:", len(self.memory))
            
            if os.path.exists(restore_path):
                with open(restore_path, "rb") as file:
                    info = pickle.load(file)
                self.steps = info["steps"]
                print(self.steps)
            else:
                print("no restore steps")

        self.summary = SummaryWriter(log_dir=reinforce_summary_path)
        
        self.trans_model_units = TransModel("trans_units", 30, 24, self.network_config, use_cuda)
        
        self.trans_model_hp = TransModel("trans_hp", 31, 1, self.network_config, use_cuda)
        
        
#         self.memory_resotre = memory_resotre
    
    def add_memory(self, pre_state, curr_state):
        self.steps += 1
        inputs = self.separate_state(pre_state, is_pre = True)
        ground_truths = self.separate_state(curr_state, is_pre = False)
        
        for input, gt in zip(inputs, ground_truths):
#             print("input")
#             print(input)
#             print("gt")
#             print(gt)
            self.memory.add(input,
                        None,
                        None,
                        gt, None)
            
        if self.steps % 10 == 0:
            self.update()
        if self.steps % 1000 == 0:
            self.save()
    
    def save(self, appendix = ""):
        info = {
            "steps": self.steps
        }
        
        print("*************saved*****************")
#         logger.info("Saving network. Found new best reward (%.2f)" % total_reward)
        self.trans_model_units.save_network(appendix = appendix)
        self.trans_model_hp.save_network(appendix = appendix)
        with open(self.network_config.network_path + "/adaptive.info", "wb") as file:
            pickle.dump(info, file, protocol=pickle.HIGHEST_PROTOCOL)
        self.memory.save(self.network_config.network_path)
        print("lenght of memeory: ", len(self.memory))

    def separate_state(self, state, is_pre):
        state = state.copy()
        
        self_buildings_top = state[1:4]
        self_buildings_bottom = state[4:7]
        
        enemy_buildings_top = state[8:11]
        enemy_buildings_bottom = state[11:14]
        
        self_units_top = state[15:27]
        self_units_bottom = state[27:39]
#         print(self_units_top)
#         print(self_units_top)
        
        self_units_top_reversed = np.concatenate((state[24:27], state[21:24], state[18:21], state[15:18]))
        self_units_bottom_reversed = np.concatenate((state[36:39], state[33:36], state[30:33], state[27:30]))
#         print(self_units_top_reversed)
#         print(self_units_bottom_reversed)
        
        enemy_units_top = state[39:51]
        enemy_units_bottom = state[51:63]
#         print(enemy_units_top)
#         print(enemy_units_bottom)
        
        enemy_units_top_reversed = np.concatenate((state[48:51], state[45:48], state[42:45], state[39:42]))
        enemy_units_bottom_reversed = np.concatenate((state[60:63], state[57:60], state[54:57], state[51:54]))
#         print(enemy_units_top)
#         print(enemy_units_bottom)
        
        self_hp_top = state[63].reshape(1)
        self_hp_bottom = state[64].reshape(1)
        
        enemy_hp_top = state[65].reshape(1)
        enemy_hp_bottom = state[66].reshape(1)
        
#         input()
        
#         print(self_buildings_top.shape, enemy_buildings_top.shape, self_units_top.shape, enemy_units_top.shape, self_hp_top.shape)
#         print(self_buildings_bottom.shape, enemy_buildings_bottom.shape, self_units_bottom.shape, enemy_units_bottom.shape, self_hp_bottom.shape)
        if is_pre:
            input_1 = np.concatenate((self_buildings_top, enemy_buildings_top, self_units_top, enemy_units_top, self_hp_top, np.array([1])))

            input_2 = np.concatenate((self_buildings_bottom, enemy_buildings_bottom, self_units_bottom, enemy_units_bottom, self_hp_bottom, np.array([2])))

            input_3 = np.concatenate((enemy_buildings_top, self_buildings_top, enemy_units_top_reversed, self_units_top_reversed, enemy_hp_top, np.array([3])))

            input_4 = np.concatenate((enemy_buildings_bottom, self_buildings_bottom, enemy_units_bottom_reversed, self_units_bottom_reversed, enemy_hp_bottom, np.array([4])))
            
            return [input_1, input_2, input_3, input_4]
        else:
            ground_truth_1 = np.concatenate((self_units_top, enemy_units_top, self_hp_top, np.array([1])))
            
            ground_truth_2 = np.concatenate((self_units_bottom, enemy_units_bottom, self_hp_bottom, np.array([2])))
            
            ground_truth_3 = np.concatenate((enemy_units_top_reversed, self_units_top_reversed, enemy_hp_top, np.array([3])))
            
            ground_truth_4 = np.concatenate((enemy_units_bottom_reversed, self_units_bottom_reversed, enemy_hp_bottom, np.array([4])))
            
            return [ground_truth_1, ground_truth_2, ground_truth_3, ground_truth_4]
    def update(self):
        if len(self.memory) < self.reinforce_config.batch_size:
            return 
        
        batch = self.memory.sample(self.reinforce_config.batch_size)
        (inputs, _, _, ground_truths, _) = batch
        
        assert np.sum(inputs[:, -1] == ground_truths[:, -1]) == len(ground_truths[:, -1]),print(inputs[:, -1], ground_truths[:, -1])
#         if self.steps == 40:
#             print(inputs[:, -1])
#             print(ground_truths[:, -1])
#             input()
        
        inputs_units = FloatTensor(inputs[:, :-2])
        inputs_hp = FloatTensor(inputs[:, :-1])
        
        gt_units = FloatTensor(ground_truths[:, : -2])
        gt_hps = FloatTensor(ground_truths[:, -2])
        
        outputs_unit = self.trans_model_units.predict_batch(inputs_units)
        outputs_hp = self.trans_model_hp.predict_batch(inputs_hp)
        
        self.trans_model_units.fit(gt_units, outputs_unit, self.steps)
        self.trans_model_hp.fit(gt_hps, outputs_hp, self.steps)