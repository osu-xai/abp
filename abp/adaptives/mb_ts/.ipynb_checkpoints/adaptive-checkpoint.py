import logging
import time
import random
import pickle
import os
from sys import maxsize

import torch
from tensorboardX import SummaryWriter
from baselines.common.schedules import LinearSchedule

from abp.utils import clear_summary_path
from abp.models import DQNModel
# TODO: Generalize it
from abp.examples.pysc2.tug_of_war.models_mb.transition_model import TransModel

logger = logging.getLogger('root')
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
IntTensor = torch.cuda.IntTensor if use_cuda else torch.IntTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


class MBTSAdaptive(object):
    """Adaptive which uses the Model base Tree search algorithm"""

    def __init__(self, name, state_length, network_config, reinforce_config, models_path, env, player = 1):
        super(MBTSAdaptive, self).__init__()
        self.name = name
        #self.choices = choices
        self.network_config = network_config
        self.reinforce_config = reinforce_config
        self.explanation = False
        self.state_length = state_length\
        
        # Global
        self.steps = 0
        self.episode = 0
        self.transition_model_HP = TransModel(state_length, 1)
        self.transition_model_unit = TransModel(state_length, 6)
        self.value_model = DQNModel(self.name + "_eval", self.network_config, use_cuda)
        self.load_model(models_path)
        self.env = env
        self.player = player
        
        self.look_forward_step = 1

        self.reset()
    def reward_func(self):
        return 
    
    
    def eval_mode(self):
        self.value_model.eval_mode()
        self.transition_model_HP.eval_mode()
        self.transition_model_unit.eval_mode()
        
    def load_model(self, models_path):
#         self.transition_model_HP.load(torch.load(models_path + 'transition_model_HP.pt').state_dict())
        # TODO: get unit transition model
#         self.transition_model_unit(torch.load(models_path + 'transition_model_unit.pt'))
        self.value_model.load_weight(torch.load(models_path + 'value_model.pt'))

    def predict(self, state):
        # Get actions of self
        all_sub_nodes = [state]
        if self.player == 1:
            fifo_self = self.env.fifo_player_1
            fifo_enemy = self.env.fifo_player_2
        else:
            fifo_self = self.env.fifo_player_2
            fifo_enemy = self.env.fifo_player_1
        
        for _ in range(self.look_forward_step)
            for n in all_sub_nodes:
                all_sub_nodes = self.expand_node(n, fifo_self, fifo_enemy)
            
            
        action, _ = self.value_model.predict(state, 0, False)
        return action
    
    def expand_node(self, state, fifo_self, fifo_enemy):
        
        self.env.get_big_A(env.denormalization(state)[env.miner_index])
       
    
    def action_ranking(self, q_state, k):
        pass
    
    def reset(self):
        self.current_reward = 0
        self.total_reward = 0

    def combine_sa(self, s, actions, fifo):
        s = np.repeat(s.reshape((1,-1)), len(actions), axis = 0)
        fifo_array = np.repeat(fifo.reshape((1,-1)), len(actions), axis = 0)
        
        actions = np.array(actions)
        s[:,:4] += actions
            
        # Get rid of the building from the candidate after_states until no exceeders to match the FIFO behavior
        for building_type in fifo:
            # Get the count of building of the candidate after_state
            count_of_bulding = s[:, :4].sum(axis = 1)
            array_of_indices_of_exceeders = count_of_bulding > self.building_limiation
            
            s[array_of_indices_of_exceeders, building_type, building_type] -= 1
            
            # Get all the fifo for each branch
            fifo_array[array_of_indices_of_exceeders, :-1] = fifo_array[array_of_indices_of_exceeders, 1:]
            fifo_array[array_of_indices_of_exceeders, 1:] = building_type
            
        s[:, self.miner_index] -= np.sum(self.maker_cost_np * actions, axis = 1) / 100
        return s, fifo_array