import logging
import time
import random
import pickle
import os
from sys import maxsize
from collections import OrderedDict

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
        self.transition_model_HP = TransModel(state_length, 2)
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
#         print("reward func")
#         print(state.shape, next_states.shape)
#         for n_s in next_states:
#             print("===================================")
#             print(state.tolist())
#             print(n_s.tolist())
# #         print(state, next_states)
#         print(next_states[:, self.index_hp] > 2000)
        next_states[next_states > 2000] = 2000
    
        rewards = (next_states - state.reshape(-1,))[:, self.index_hp]
        rewards[rewards > 0] = 1
        rewards[:, 1] *= -1
        rewards = np.sum(rewards, axis = 1)
#         print(rewards)
#         input()
        return rewards
    
    
    def eval_mode(self):
        self.value_model.eval_mode()
        self.transition_model_HP.eval_mode()
        self.transition_model_unit.eval_mode()
        
    def load_model(self, models_path):
        HP_state_dict = torch.load(models_path + 'transition_model_HP.pt')
        unit_state_dict = torch.load(models_path + 'transition_model_unit.pt')
#         print(HP_state_dict.model)
        new_HP_state_dict = OrderedDict()
        new_unit_state_dict = OrderedDict()
        
        for old_key_value_hp, old_key_value_unit in zip(list(HP_state_dict.items()), list(unit_state_dict.items())):
            new_key_hp, new_value_hp = "module." + old_key_value_hp[0], old_key_value_hp[1]
            new_key_unit, new_value_unit = "module." + old_key_value_unit[0], old_key_value_unit[1]
#             print(new_key_hp, new_key_unit)
#             print(old_key_hp, old_key_unit)
            new_HP_state_dict[new_key_hp] = new_value_hp
            new_unit_state_dict[new_key_unit] = new_value_unit
        
        
        self.transition_model_HP.load_weight(new_HP_state_dict)
        # TODO: get unit transition model
        self.transition_model_unit.load_weight(new_unit_state_dict)
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
                next_states, next_fifo_self, length_enemy_action = self.expand_node(n, minerals_enemy, fifo_self, fifo_enemy)
                if i == (self.look_forward_step - 1):
                    next_states = self.same_self_action_block(next_states, length_enemy_action)
                    children = self.same_self_action_block(np.array(n.children), length_enemy_action)
                    leaf_node_states.append(next_states)
                    leaf_fifo.append(next_fifo_self)
                    leaf_node.append(children)
                
        
#         print(len(leaf_node_states[0]), len(leaf_fifo[0]), len(leaf_node[0]))
#         input()
        if self.look_forward_step == 0:
            self.rollout_root([parents[0].state], parents, [fifo_self])
        else:
            for lns, ln, ff in zip(leaf_node_states, leaf_node, leaf_fifo):
                self.rollout(lns, ln, ff)
        
#         print(root.best_reward)
#         action, _ = self.value_model.predict(state, 0, False)
#         print(root.best_action)
        return root.best_action

    def same_self_action_block(self, states_or_nodes, length_enemy_action):
        return np.array(np.split(states_or_nodes, length_enemy_action))
    
    def rollout(self, states, nodes, ffs):
        all_min_q_value = []
        for s_block, n_block, ff in zip(states, nodes, ffs):
#             print(s.tolist(),n.parent.best_reward,ff)
#             input()
            s_b = []
            for s in s_block:
                actions_self = self.env.get_big_A(s[self.env.miner_index])
                com_s, _ = self.combine_sa(s, actions_self, ff, is_enemy = False)
    #             for cs in com_s:
    #                 print(cs.tolist())
    #             input()
    #             com_s_old, _ = self.combine_sa_old(s, actions_self, ff, is_enemy = False)
    #             assert sum(sum(com_s_old == com_s)) == com_s_old.shape[0] * com_s_old.shape[1], print(com_s_old == com_s)
                com_s = self.env.normalization(com_s)
                s_b.append(com_s)
            s_b = np.vstack(s_b)
#             print(s_b.shape)
#             input()
            q_values_block = FloatTensor(self.value_model.predict_batch(Tensor(s_b))[1]).view(-1)
#             print(q_values)
#             input()
            min_q_value, _ = q_values_block.min(0)
            all_min_q_value.append(min_q_value)
#         print(all_min_q_value)
        max_q_value, choice = FloatTensor(all_min_q_value).max(0)
        if nodes[0][0].parent is not None:
            parent = nodes[0][0].parent
        else:
            parent = nodes[0][0]
        parent.best_reward = parent.reward + max_q_value
        parent.best_action = self.env.get_big_A(parent.state[self.env.miner_index])[choice]
        self.reward_brack_prop(parent)
#         print("mbts:")
#         print(parent.best_reward)
#         print(parent.best_action)
#         input()
        
    def rollout_root(self, states, nodes, ffs):
        for s, n, ff in zip(states, nodes, ffs):
            actions_self = self.env.get_big_A(s[self.env.miner_index])
            com_s, _ = self.combine_sa(s, actions_self, ff, is_enemy = False)
            com_s = self.env.normalization(com_s)
            q_values = FloatTensor(self.value_model.predict_batch(Tensor(com_s))[1]).view(-1)
            max_q_value, choice = q_values.max(0)
            n.best_reward = n.reward + max_q_value
            n.best_action = actions_self[choice]
#             print("sadq:")
#             print(n.reward + max_q_value)
#             print(actions_self[choice])
#             input()
    def normalization(self, state):
        return state / self.normalization_array
    
    def expand_node(self, parent, mineral_enemy, fifo_self, fifo_enemy):
        # TODO: check the state change or not ,if yes deepcopy for the reward func state
        state = deepcopy(parent.state)
        parent_name = parent.name
        actions_self = self.env.get_big_A(state[self.env.miner_index])
        actions_enemy = self.env.get_big_A(mineral_enemy)
        
        after_states, after_fifo_self, after_state_actions_self = self.get_after_states(state, actions_self, actions_enemy, fifo_self, fifo_enemy)
        next_states = self.get_next_states(after_states)
        
        rewards = self.reward_func(state, next_states)
        
        all_sub_nodes = []
        best_reward = float('-inf')
        best_node = None
        best_action = None
#         print(after_state_actions_self)
        for i, (n_s, reward, action) in enumerate(zip(next_states, rewards, after_state_actions_self)):
#             print(n_s.tolist(), reward, action)
#             input()
            child = Node(parent_name + '_' + str(i + 1), n_s, reward = reward, parent = parent, parent_action = action)
            parent.add_child(child, action)
            
            if best_reward < reward:
                best_reward = reward
                best_node = child
                best_action = action
        parent.best_action = best_action
        self.reward_brack_prop(best_node)
    
        return next_states, after_fifo_self, len(actions_self)
    
    def reward_brack_prop(self, node):
        if node.name == "root":
            return
        parent = node.parent
        if node.best_reward > parent.best_reward:
            parent.best_child = node
            parent.best_reward = node.best_reward
            parent.best_action = node.parent_action
            self.reward_brack_prop(parent)
        return
    
    def action_ranking(self, q_state, k):
        # TODO
        pass
    
    def reset(self):
        self.current_reward = 0
        self.total_reward = 0

    def get_next_states(self, after_state):
        next_HPs = self.transition_model_HP.predict_batch(FloatTensor(self.normalization(after_state)))
        next_units = self.transition_model_unit.predict_batch(FloatTensor(self.normalization(after_state)))
        
        after_state[:, self.index_hp] = next_HPs.cpu().detach().numpy()
        after_state[:, self.index_units] = next_units.round().cpu().detach().numpy()
        
        return after_state
        
    def get_after_states(self, state, actions_self, actions_enemy, fifo_self, fifo_enemy):
        # Faster combination way
        
        after_states_self, after_fifo_self = self.combine_sa(state, actions_self, fifo_self, is_enemy = False)
#         print(len(actions_self), len(after_fifo_self))
#         after_states_self = self.imply_mineral_by_action(after_states_self, actions_self)
#         for af, ff in zip(after_states_self, after_fifo_self):
#             print(af.tolist(), ff)
        after_states = np.zeros((len(actions_self) * len(actions_enemy), after_states_self.shape[1]))
#         print(after_states.shape)
#         idx = 0
        after_state_actions_self = np.zeros((len(actions_self) * len(actions_enemy), actions_self.shape[1]))
#         after_state_fifo_self = []
        for i, a_s_s in enumerate(after_states_self):
            a_s, _ = self.combine_sa(a_s_s, actions_enemy, fifo_enemy, is_enemy = True)
#             print(a_s.shape)
            after_states[i * len(actions_enemy) : (i + 1)* len(actions_enemy)] = a_s
    
            after_state_actions_self[i * len(actions_enemy) : (i + 1)* len(actions_enemy)] = np.repeat(actions_self[i].reshape((1,-1)), len(actions_enemy), axis = 0).copy()
#             for _ in range(len(actions_enemy)):
#                 after_state_fifo_self.append(deepcopy(after_fifo_self[i]))
#         print(after_states[:, building_types['Pylon']])
#         print("*********")
#         print(len(after_states), len(after_fifo_self), len(actions_enemy))
        after_states[:, self.env.miner_index] += after_states[:, building_types['Pylon']] * 50 + 100
#             idx += 1
#         print(idx)
#         for a_s in after_states:
#             print(a_s.tolist())
        
        return after_states, after_fifo_self, after_state_actions_self
        
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

