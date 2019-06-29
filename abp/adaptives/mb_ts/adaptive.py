import logging
import time
import random
import pickle
import os
from sys import maxsize

import torch
from tensorboardX import SummaryWriter
from baselines.common.schedules import LinearSchedule
import numpy as np
from copy import deepcopy

from abp.utils import clear_summary_path
from abp.models import DQNModel
# TODO: Generalize it
from abp.examples.pysc2.tug_of_war.models_mb.transition_model import TransModel
from abp.utils.search_tree import Node

logger = logging.getLogger('root')
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
IntTensor = torch.cuda.IntTensor if use_cuda else torch.IntTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

building_types = {
    'Marine': 0,
    'Viking': 1,
    'Colossus': 2,
    'Pylon': 3
}
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
        
        self.index_hp = np.array([4, 9])
        self.index_units = np.array(range(11, 17))
        
        self.look_forward_step = 1
        
        # Generalize it 
        self.load_model(models_path)
        self.eval_mode()
        self.normalization_array = np.array([30, 30, 30, 30, 2000,
                                             30, 30, 30, 30, 2000,
                                             1500, 60, 60, 60, 60, 60, 60])

        self.reset()
        
    def reward_func(self, state, next_states):
        print("reward func")
        for n_s in next_states:
            print("===================================")
            print(state.tolist())
            print(n_s.tolist())
#         print(state, next_states)
        input()
        return 
    
    
    def eval_mode(self):
        self.value_model.eval_mode()
        self.transition_model_HP.eval_mode()
        self.transition_model_unit.eval_mode()
        
    def load_model(self, models_path):
#         print(models_path)
#         self.transition_model_HP.load_weight(torch.load(models_path + 'transition_model_HP.pt').state_dict())
        # TODO: get unit transition model
        self.transition_model_unit.load_weight(torch.load(models_path + 'transition_model_unit.pt'))
        self.value_model.load_weight(torch.load(models_path + 'value_model.pt'))

    def predict(self, state, minerals_enemy):
        # Get actions of self
        root = Node('root', state)
        parents = [root]
        if self.player == 1:
            fifo_self = self.env.fifo_player_1
            fifo_enemy = self.env.fifo_player_2
        else:
            fifo_self = self.env.fifo_player_2
            fifo_enemy = self.env.fifo_player_1
        
        leaf_node = []
        leaf_node_states = []
        leaf_fifo = []
        for i in range(self.look_forward_step):
            for n in parents:
                next_states, next_fifo_self = self.expand_node(n, minerals_enemy, fifo_self, fifo_enemy)
                if i == (self.look_forward_step - 1):
                    leaf_node_states.append(next_states)
                    leaf_fifo.append(next_fifo_self)
                    leaf_node.append(n.children)
                    
        if self.look_forward_step == 0:
            leaf_node.append(parents)
            leaf_node_states.append([parents[0].state])
            leaf_fifo.append([fifo_self])
            
        for lns, ln, ff in zip(leaf_node_states, leaf_node, leaf_fifo):
            self.rollout(lns, ln, ff)
            
#         action, _ = self.value_model.predict(state, 0, False)
#         print(root.best_action)
        return root.best_action
    
    def rollout(self, states, nodes, ffs):
        for s, n, ff in zip(states, nodes, ffs):
            actions_self = self.env.get_big_A(s[self.env.miner_index])
            com_s, _ = self.combine_sa(s, actions_self, ff, is_enemy = False)
#             com_s_old, _ = self.combine_sa_old(s, actions_self, ff, is_enemy = False)
#             assert sum(sum(com_s_old == com_s)) == com_s_old.shape[0] * com_s_old.shape[1], print(com_s_old == com_s)
            com_s = self.env.normalization(com_s)
#             print(s)
#             print()
#             for cs in com_s:
#                 print(cs.tolist())
#             print()
#             print(actions_self)
#             input()
            q_values = FloatTensor(self.value_model.predict_batch(Tensor(com_s))[1]).view(-1)
#             print(q_values)
#             input()
            max_q_value, choice = q_values.max(0)
            n.best_reward = n.reward + max_q_value
            n.best_action = actions_self[choice]
            self.reward_brack_prop(n)
        
    def expand_node(self, parent, mineral_enemy, fifo_self, fifo_enemy):
        # TODO: check the state change or not ,if yes deepcopy for the reward func state
        state = deepcopy(parent.state)
        parent_name = parent.name
        actions_self = self.env.get_big_A(state[self.env.miner_index])
        actions_enemy = self.env.get_big_A(mineral_enemy)
        
        after_states, after_fifo_self = self.get_after_states(state, actions_self, actions_enemy, fifo_self, fifo_enemy)
        next_states = self.get_next_states(after_states)
        
        rewards = self.reward_func(state, next_states)
        
        all_sub_nodes = []
        best_reward = float('-inf')
        best_node = None
        best_action = None
        for i, n_s, reward, action in enumerate(zip(next_states, rewards, actions_self)):
            child = Node(parent_name + '_' + str(i + 1)), n_s, reward, parent
            parent.add_child(child, action)
            
            if best_reward < reward:
                best_reward = reward
                best_node = child
                best_action = action
            
        self.reward_brack_prop(best_node)
    
        return next_states, after_fifo_self
    
    def reward_brack_prop(self, node):
        if node.name == "root":
            return
        parent = node.parent
        if node.best_reward > parent.best_reward:
            parent.best_child = node
            parent.best_reward = node.best_reward
            self.reward_brack_prop(parent)
            return
    
    def action_ranking(self, q_state, k):
        # TODO
        pass
    
    def reset(self):
        self.current_reward = 0
        self.total_reward = 0

    def get_next_states(self, after_state):
        next_HPs = self.transition_model_HP.predict_batch(FloatTensor(after_state))
        next_units = self.transition_model_unit.predict_batch(FloatTensor(after_state))
        
        after_state[:, self.index_hp] = next_HPs.cpu().detach().numpy()
        after_state[:, self.index_units] = next_units.cpu().detach().numpy()
        
        return after_state
        
    def get_after_states(self, state, actions_self, actions_enemy, fifo_self, fifo_enemy):
        # Faster combination way
        after_states_self, after_fifo_self = self.combine_sa(state, actions_self, fifo_self, is_enemy = False)
#         after_states_self = self.imply_mineral_by_action(after_states_self, actions_self)
#         for af, ff in zip(after_states_self, after_fifo_self):
#             print(af.tolist(), ff)
        after_states = np.zeros((len(actions_self) * len(actions_enemy), after_states_self.shape[1]))
#         print(after_states.shape)
#         idx = 0
        for i, a_s_s in enumerate(after_states_self):
            a_s, _ = self.combine_sa(a_s_s, actions_enemy, fifo_enemy, is_enemy = True)
#             print(a_s.shape)
            after_states[i * len(actions_enemy) : (i + 1)* len(actions_enemy)] = a_s
#         print(after_states[:, building_types['Pylon']])
        after_states[:, self.env.miner_index] += after_states[:, building_types['Pylon']] * 50 + 100
#             idx += 1
#         print(idx)
#         for a_s in after_states:
#             print(a_s.tolist())
        return after_states, after_fifo_self
        
#     def combine_sa_old(self, de_s, actions, fifo, is_enemy):
#         # Change that if the index is changed, generalize it later
#         if not is_enemy:
#             building_index = range(0, 4)
#         else:
#             building_index = range(5, 9)
        
#         fifo = np.array(fifo)
#         s = np.repeat(de_s.reshape((1,-1)), len(actions), axis = 0)
#         fifo_array = np.repeat(fifo.reshape((1,-1)), len(actions), axis = 0)
        
#         actions = np.array(actions)
#         s[:,building_index] += actions
            
#         # Get rid of the building from the candidate after_states until no exceeders to match the FIFO behavior
#         for building_type in fifo:
#             # Get the count of building of the candidate after_state
#             count_of_bulding = s[:, building_index].sum(axis = 1)
#             array_of_indices_of_exceeders = count_of_bulding > self.env.building_limiation
            
#             if sum(array_of_indices_of_exceeders) <= 0:
#                 break
# #             print(s.shape)
#             s[array_of_indices_of_exceeders, building_type] -= 1
            
#             # Get all the fifo for each branch
#             fifo_array[array_of_indices_of_exceeders, :-1] = fifo_array[array_of_indices_of_exceeders, 1:]
#             fifo_array[array_of_indices_of_exceeders, 1:] = building_type
            
#         if not is_enemy:    
#             s[:, self.env.miner_index] -= np.sum(self.env.maker_cost_np * actions, axis = 1)
#         return s, fifo_array

    def combine_sa(self, de_s, actions, fifo, is_enemy):
        if not is_enemy:
            building_index = list(range(0, 4))
        else:
            building_index = list(range(5, 9))
        fifo_list = []
        for _ in range(len(actions)):
            fifo_list.append(deepcopy(fifo))
        s = np.repeat(de_s.reshape((1,-1)), len(actions), axis = 0)
        actions = actions.reshape(-1, 4)
        for idx_a, action in enumerate(actions):
#             print(action)
            for a, num in enumerate(action):
                for _ in range(int(num)):
                    s[idx_a][building_index[a]] += 1
                    fifo_list[idx_a].append(building_index[a])
                    if len(fifo_list[idx_a]) > 30:
                        s[idx_a][building_index[fifo_list[idx_a][0]]] -= 1
                        del fifo_list[idx_a][0]
        if not is_enemy:    
            s[:, self.env.miner_index] -= np.sum(self.env.maker_cost_np * actions, axis = 1)
        return s, fifo_list
    
    def imply_mineral_by_action(self, mineral, action):
        mineral -= np.sum(self.env.maker_cost_np * action)
        return mineral
        
    def imply_after_mineral(self, state):
        state[env.miner_index] += state[building_types['Pylon']] * 50 + 100
        return state

