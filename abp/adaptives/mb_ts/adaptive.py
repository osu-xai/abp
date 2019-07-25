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
from abp.configs import NetworkConfig, ReinforceConfig, EvaluationConfig
from abp.models import TransModel
from tqdm import tqdm

logger = logging.getLogger('root')
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
IntTensor = torch.cuda.IntTensor if use_cuda else torch.IntTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor
np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)
class MBTSAdaptive(object):
    """Adaptive which uses the Model base Tree search algorithm"""

    def __init__(self, name, state_length, network_config, reinforce_config, 
                 models_path, env, depth, action_ranking, player = 1):
        super(MBTSAdaptive, self).__init__()
        self.name = name
        #self.choices = choices
        self.network_config = network_config
        self.reinforce_config = reinforce_config
        self.explanation = False
        self.state_length = state_length
        
        self.init_network()
        self.load_model(models_path)
        # Global
        self.steps = 0
        self.episode = 0
        # TODO: Generalize it
        self.env = env
        self.player = player
        
        self.index_hp = LongTensor(range(27, 31))
        self.index_units = LongTensor(range(15, 27))
        self.index_building_self = LongTensor(range(1, 8))
        self.index_building_enemy = LongTensor(range(8, 15))
        self.pylon_index_self = self.env.pylon_index
        self.pylon_index_enemy = 14
        self.mineral_index_self = self.env.miner_index
        self.mineral_index_enemy = state_length
        self.maker_cost_np = FloatTensor(self.env.maker_cost_np)
        self.index_waves = 31
        
        self.norm_vector_vf = FloatTensor([700, 50, 40, 20, 50, 40, 20, 3,
                                    50, 40, 20, 50, 40, 20, 3,
                                    50, 40, 20, 50, 40, 20, 
                                    50, 40, 20, 50, 40, 20,
                                    2000, 2000, 2000, 2000, 40])
        
        self.norm_vector_unitandhp = FloatTensor([1500, # p1 minerals
                        30, 30, 10, 30, 30, 10, 3, # p1 top and bottom lane buildings
                        30, 30, 10, 30, 30, 10, 3, # p2 top and bottom lane buildings
                        30, 30, 10, 30, 30, 10, # p1 top and bottom lane units
                        30, 30, 10, 30, 30, 10, # p1 top and bottom lane units
                        2000, 2000, 2000, 2000, 40])
        self.look_forward_step = depth
        self.ranking_topk = action_ranking#float('inf')
        
        # Generalize it 
        self.eval_mode()

#         self.reset()
        
        
    def init_network(self):
        network_trans_unit_path = "./tasks/tug_of_war/trans/unit/v1/network.yml"
        network_trans_hp_path = "./tasks/tug_of_war/trans/health/v1/network.yml"
        network_trans_unit = NetworkConfig.load_from_yaml(network_trans_unit_path)
        network_trans_hp = NetworkConfig.load_from_yaml(network_trans_hp_path)
        
        self.transition_model_HP = TransModel("TugOfWar2lNexusHealth",network_trans_hp, use_cuda)
        
        self.transition_model_unit = TransModel("TugOfWar2lNexusUnit",network_trans_unit, use_cuda)
        
        self.value_model = DQNModel(self.name + "_eval", self.network_config, use_cuda)
        self.q_model = DQNModel(self.name + "_eval", self.network_config, use_cuda)
        
    def load_model(self, models_path):
        HP_state_dict = torch.load(models_path + 'transition_model_hp.pt')
        unit_state_dict = torch.load(models_path + 'transition_model_unit.pt')
        value_state_dict = torch.load(models_path + 'value_model.pt')

#         print(HP_state_dict)
        new_HP_state_dict = OrderedDict()
        new_unit_state_dict = OrderedDict()
        new_value_state_dict = OrderedDict()
        
        the_HP_weight = list(HP_state_dict.values())
        the_unit_weight = list(unit_state_dict.values())
        the_value_weight = list(value_state_dict.values())
        
        the_HP_keys = list(self.transition_model_HP.model.state_dict().keys())
        the_unit_keys = list(self.transition_model_unit.model.state_dict().keys())
        the_value_keys = list(self.value_model.model.state_dict().keys())
        
#         print(the_HP_keys)
#         print(the_unit_keys)
#         print(the_value_keys)
#         print("*************")
        for i in range(len(the_HP_weight)):
            new_HP_state_dict[the_HP_keys[i]] = the_HP_weight[i]
        for i in range(len(the_HP_weight)):
            new_unit_state_dict[the_unit_keys[i]] = the_unit_weight[i]
        for i in range(len(the_HP_weight)):
            new_value_state_dict[the_value_keys[i]] = the_value_weight[i]
            
        self.transition_model_HP.load_weight(new_HP_state_dict)
        self.transition_model_unit.load_weight(new_unit_state_dict)
#         self.value_model.load_weight(new_value_state_dict)
        self.q_model.load_weight(torch.load(models_path + 'q_model.pt'))
        
    def player_1_win_condition(self, state_1_T_hp, state_1_B_hp, state_2_T_hp, state_2_B_hp):
        if min(state_1_T_hp, state_1_B_hp) == min(state_2_T_hp, state_2_B_hp):
            if state_1_T_hp + state_1_B_hp == state_2_T_hp + state_2_B_hp:
                return 0
            elif state_1_T_hp + state_1_B_hp > state_2_T_hp + state_2_B_hp:
                return 1
            else:
                return -1
        else:
            if min(state_1_T_hp, state_1_B_hp) > min(state_2_T_hp, state_2_B_hp):
                return 1
            else:
                return -1
            
    def reward_func(self, state, next_states):
        # TODO
        rewards = FloatTensor((next_states - state.reshape(-1,))[:, self.index_hp].clone())
        
#         print(rewards)
        rewards[rewards > 0] = 0
#         print(rewards)
        rewards[:, 2:] *= -1
#         print(rewards)
#         print(rewards)
        rewards = torch.sum(rewards, dim = 1)
        min_s, index = next_states[:, self.index_hp].min(1)
        win = ((min_s < 10) & (index > 1)).float()
        lose = ((min_s < 10) & (index <= 1)).float()
        
        rewards += win * 10000
        rewards -= lose * 10000
#         if sum(win) > 0 or sum(lose) > 0:
#             print(min_s, index)
#             print(win, lose)
#             print(rewards)
#             input()
        return rewards
    
    
    def eval_mode(self):
#         self.value_model.eval_mode()
        self.transition_model_HP.eval_mode()
        self.transition_model_unit.eval_mode()
        self.q_model.eval_mode()

        
    def minimax(self, state, depth = 1, previous_reward = 0.0):
#         print("20---------------------------")
#         print(state)
        actions_self = self.env.get_big_A(state[self.mineral_index_self].item(), state[self.pylon_index_self].item())
#         print("3---------------------------")
#         print(actions_self)
        com_states = self.combine_sa(state, actions_self, is_enemy = False)
#         print("4---------------------------")
#         print(com_states)
        top_k_action_self = self.action_ranking(com_states, actions_self)
#         print("5---------------------------")
#         print(top_k_action_self)
        
        actions_enemy = self.env.get_big_A(state[self.mineral_index_enemy].item(), state[self.pylon_index_enemy].item())
#         print("6---------------------------")
#         print(state[self.mineral_index_enemy].item(), actions_enemy)
        # action ranking here
#         print(state)
        com_states = self.combine_sa(self.switch_state_to_enemy(state), actions_enemy, is_enemy = False)
#         print(state)
#         print("7---------------------------")
#         print(com_states)
        top_k_action_enemy = self.action_ranking(com_states, actions_enemy)
#         print("8---------------------------")
#         print(top_k_action_enemy)
        
        max_value = float("-inf")
        for a_self in tqdm(top_k_action_self, desc = "Making decision depth = " + str(depth)):
#             print("9---------------------------")
#             print(a_self)
#             print("10---------------------------")
#             print(top_k_action_enemy)
            next_states = self.get_next_states(state, a_self, top_k_action_enemy)
#             print("11---------------------------")
#             print(next_states)
            next_reward = self.reward_func(state, next_states)
#             print("12---------------------------")
#             print(next_reward)
        
            if depth == self.look_forward_step:
                min_values = self.rollout(next_states)
            else:
                min_values = FloatTensor(np.zeros(len(next_states)))
                for i, (n_s, n_r) in enumerate(zip(next_states, next_reward)):
                    if abs(n_r) >= 10000:
                        continue
                    min_values[i], _ = self.minimax(n_s, depth = depth + 1, previous_reward = previous_reward + n_r)
            
#             print("13---------------------------")
#             print(min_values)
            min_values = min_values.view(-1)
            previous_reward = FloatTensor([previous_reward])
#             print("21---------------------------")
#             print(next_reward.is_cuda, previous_reward.is_cuda, previous_reward.is_cuda)
#             print(min_values.size(), next_reward.size(), previous_reward)
            min_values += (next_reward + previous_reward)
#             print("14---------------------------")
#             print(next_reward)
#             print("15---------------------------")
#             print(previous_reward)
#             print("16---------------------------")
#             print(min_values)
            min_value = min_values.min(0)[0].item()
#             print("17---------------------------")
#             print(min_value)
            if max_value < min_value:
                max_value = min_value
                best_action = a_self
#             print("18---------------------------")
#             print(max_value)
#             print("19---------------------------")
#             print(best_action)
#             input()
        return max_value, best_action

    def predict(self, state, minerals_enemy):
#         print("1---------------------------")
#         print(state)
        state = FloatTensor(np.append(state, minerals_enemy))
#         print("2---------------------------")
#         print(state)
        if self.look_forward_step > 0:
            max_value, best_action = self.minimax(state)
        else:
            pass
#         print(state)
#         input()

        return best_action
    
    def normalization(self, state, is_vf = False):
        if not is_vf:
            return state[:, :-1] / self.norm_vector_unitandhp
        else:
            return state[:, :-1] / self.norm_vector_vf
    
    def rollout(self, states):
#         return self.value_model.predict_batch(FloatTensor(self.normalization(states, is_vf = True)))[1]

        values = FloatTensor(np.zeros(len(states)))
        with torch.no_grad():
            for i, state in enumerate(states):
                actions_self = self.env.get_big_A(state[self.mineral_index_self].item(), state[self.pylon_index_self].item())

                com_states = self.combine_sa(state, actions_self, is_enemy = False)

                values[i] = self.q_model.predict_batch(FloatTensor(self.normalization(states, is_vf = True)))[1].max(0)[0]
        return values
    
    def action_ranking(self, after_states, action, ):
#         action = FloatTensor(action)
#         print(len(after_states), len(action))
        if self.ranking_topk >= len(after_states):
            return action[np.array(range(len(after_states)))]
        with torch.no_grad():
            q_values = self.value_model.predict_batch(FloatTensor(self.normalization(after_states, is_vf = True)))[1]
            q_values = q_values.view(-1)
            topk_q, indices = torch.topk(q_values, self.ranking_topk)
        
#         print(topk_q)
#         print(indices)
#         input()
        return action[indices.cpu().clone().numpy()]
    
    def switch_state_to_enemy(self, states):
        enemy_states = states.clone()
#         print(enemy_states)
        mineral_index_self = LongTensor(range(0, 1))
        mineral_index_enemy = LongTensor(range(32, 33))
        buliding_index_self = self.index_building_self
        buliding_index_enemy = self.index_building_enemy
        unit_index_self = LongTensor(range(15, 21))
        unit_index_enemy = LongTensor(range(21, 27))
        index_hp_self = LongTensor(range(27, 29))
        index_hp_enemy = LongTensor(range(29, 31))
        
        enemy_states[mineral_index_self], enemy_states[mineral_index_enemy], \
        enemy_states[buliding_index_self], enemy_states[buliding_index_enemy], \
        enemy_states[unit_index_self], enemy_states[unit_index_enemy], \
        enemy_states[index_hp_self], enemy_states[index_hp_enemy] \
        = \
        enemy_states[mineral_index_enemy], enemy_states[mineral_index_self], \
        enemy_states[buliding_index_enemy], enemy_states[buliding_index_self], \
        enemy_states[unit_index_enemy], enemy_states[unit_index_self], \
        enemy_states[index_hp_enemy], enemy_states[index_hp_self]
        
#         print(enemy_states)
        
#         input()
        return enemy_states
    
    def get_next_states(self, state, a_self, top_k_action_enemy):
        after_states = self.get_after_states(state, a_self, top_k_action_enemy)
        
        next_states = after_states.clone()
#         print(next_states.shape)
        with torch.no_grad():
            next_HPs = self.transition_model_HP.predict_batch(FloatTensor(self.normalization(next_states)))
            next_units = self.transition_model_unit.predict_batch(FloatTensor(self.normalization(next_states)))
#         print(next_HPs.shape)
#         print(next_units.shape)
        next_states[:, self.index_hp] = next_HPs
        next_states[:, self.index_units] = next_units.round()
        next_states[:, self.index_waves] += 1

        return next_states

    def get_after_states(self, state, a_self, top_k_action_enemy):
        # Faster combination way
        after_states_self = self.combine_sa(state, np.array(a_self).reshape(-1, 7), False)
        after_states = self.combine_sa(after_states_self, top_k_action_enemy, True)
        
        after_states[:, self.mineral_index_self] += after_states[:, self.pylon_index_self] * 75 + 100
        
        after_states[:, self.mineral_index_enemy] += after_states[:, self.pylon_index_enemy] * 75 + 100
        
        return after_states

    def combine_sa(self, state, actions, is_enemy):
        if not is_enemy:
            building_index = self.index_building_self
            mineral_index = self.mineral_index_self
            pylon_index = self.pylon_index_self
        else:
            building_index = self.index_building_enemy
            mineral_index = self.mineral_index_enemy
            pylon_index =  self.pylon_index_enemy
            
        state = state.clone()
        com_state = state.repeat((len(actions), 1))
        actions = FloatTensor(actions)
        
#         print(com_state.shape)
#         print(actions.shape)
        com_state[:,building_index] += actions
        
        com_state[:, mineral_index] -= torch.sum(self.maker_cost_np * actions[:, :-1], dim = 1)

        index_has_pylon = actions[:, -1] > 0
        num_of_pylon = com_state[index_has_pylon, pylon_index]
        com_state[index_has_pylon, mineral_index] -= (self.env.pylon_cost + (num_of_pylon - 1) * 100)
        
        
        assert torch.sum(com_state[:, mineral_index] >= 0) == len(com_state), print(com_state)

        return com_state
    
    
    
    
#     def predict(self, state, minerals_enemy):
#         # Get actions of self
#         root = Node('root', state)
#         parents = [root]
        
#         leaf_node = []
#         leaf_node_states = []
#         leaf_fifo = []
        
        
# #         for i in range(self.look_forward_step):
# #             for n in parents:
# #                 next_states, next_fifo_self, length_enemy_action = self.expand_node(n, minerals_enemy, fifo_self, fifo_enemy)
# #                 if i == (self.look_forward_step - 1):
# #                     next_states = self.same_self_action_block(next_states, length_enemy_action)
# #                     children = self.same_self_action_block(np.array(n.children), length_enemy_action)
# #                     leaf_node_states.append(next_states)
# #                     leaf_fifo.append(next_fifo_self)
# #                     leaf_node.append(children)
                
        
# #         print(len(leaf_node_states[0]), len(leaf_fifo[0]), len(leaf_node[0]))
# #         input()
# #         if self.look_forward_step == 0:
# #             self.rollout_root([parents[0].state], parents, [fifo_self])
# #         else:
# #             for lns, ln, ff in zip(leaf_node_states, leaf_node, leaf_fifo):
# #                 self.rollout(lns, ln, ff)
        
# #         print(root.best_reward)
# #         action, _ = self.value_model.predict(state, 0, False)
# #         print(root.best_action)
#         return root.best_action

#     def same_self_action_block(self, states_or_nodes, length_enemy_action):
#         return np.array(np.split(states_or_nodes, length_enemy_action))
    
#     def rollout(self, states, nodes, ffs):
#         all_min_q_value = []
#         for s_block, n_block, ff in zip(states, nodes, ffs):
# #             print(s.tolist(),n.parent.best_reward,ff)
# #             input()
#             s_b = []
#             for s in s_block:
#                 actions_self = self.env.get_big_A(s[self.env.miner_index])
#                 com_s, _ = self.combine_sa(s, actions_self, ff, is_enemy = False)
#     #             for cs in com_s:
#     #                 print(cs.tolist())
#     #             input()
#     #             com_s_old, _ = self.combine_sa_old(s, actions_self, ff, is_enemy = False)
#     #             assert sum(sum(com_s_old == com_s)) == com_s_old.shape[0] * com_s_old.shape[1], print(com_s_old == com_s)
#                 com_s = self.env.normalization(com_s)
#                 s_b.append(com_s)
#             s_b = np.vstack(s_b)
# #             print(s_b.shape)
# #             input()
#             q_values_block = FloatTensor(self.value_model.predict_batch(Tensor(s_b))[1]).view(-1)
# #             print(q_values)
# #             input()
#             min_q_value, _ = q_values_block.min(0)
#             all_min_q_value.append(min_q_value)
# #         print(all_min_q_value)
#         max_q_value, choice = FloatTensor(all_min_q_value).max(0)
#         if nodes[0][0].parent is not None:
#             parent = nodes[0][0].parent
#         else:
#             parent = nodes[0][0]
#         parent.best_reward = parent.reward + max_q_value
#         parent.best_action = self.env.get_big_A(parent.state[self.env.miner_index])[choice]
#         self.reward_brack_prop(parent)
# #         print("mbts:")
# #         print(parent.best_reward)
# #         print(parent.best_action)
# #         input()
        
#     def rollout_root(self, states, nodes, ffs):
#         for s, n, ff in zip(states, nodes, ffs):
#             actions_self = self.env.get_big_A(s[self.env.miner_index])
#             com_s, _ = self.combine_sa(s, actions_self, ff, is_enemy = False)
#             com_s = self.env.normalization(com_s)
#             q_values = FloatTensor(self.value_model.predict_batch(Tensor(com_s))[1]).view(-1)
#             max_q_value, choice = q_values.max(0)
#             n.best_reward = n.reward + max_q_value
#             n.best_action = actions_self[choice]
# #             print("sadq:")
# #             print(n.reward + max_q_value)
# #             print(actions_self[choice])
# #             input()
#     def normalization(self, state):
#         return state / self.normalization_array
    
#     def expand_node(self, parent, mineral_enemy, fifo_self, fifo_enemy):
#         # TODO: check the state change or not ,if yes deepcopy for the reward func state
#         state = deepcopy(parent.state)
#         parent_name = parent.name
#         actions_self = self.env.get_big_A(state[self.env.miner_index])
#         actions_enemy = self.env.get_big_A(mineral_enemy)
        
#         after_states, after_fifo_self, after_state_actions_self = self.get_after_states(state, actions_self, actions_enemy, fifo_self, fifo_enemy)
#         next_states = self.get_next_states(after_states)
        
#         rewards = self.reward_func(state, next_states)
        
#         all_sub_nodes = []
#         best_reward = float('-inf')
#         best_node = None
#         best_action = None
# #         print(after_state_actions_self)
#         for i, (n_s, reward, action) in enumerate(zip(next_states, rewards, after_state_actions_self)):
# #             print(n_s.tolist(), reward, action)
# #             input()
#             child = Node(parent_name + '_' + str(i + 1), n_s, reward = reward, parent = parent, parent_action = action)
#             parent.add_child(child, action) 
            
#             if best_reward < reward:
#                 best_reward = reward
#                 best_node = child
#                 best_action = action
#         parent.best_action = best_action
#         self.reward_brack_prop(best_node)
    
#         return next_states, after_fifo_self, len(actions_self)
    
#     def reward_brack_prop(self, node):
#         if node.name == "root":
#             return
#         parent = node.parent
#         if node.best_reward > parent.best_reward:
#             parent.best_child = node
#             parent.best_reward = node.best_reward
#             parent.best_action = node.parent_action
#             self.reward_brack_prop(parent)
#         return
    
#     def action_ranking(self, q_state, k = 10):
#         # TODO
#         return np.array(range(0, len(q_state)))
    
#     def reset(self):
#         self.current_reward = 0
#         self.total_reward = 0

#     def get_next_states(self, after_state):
#         next_HPs = self.transition_model_HP.predict_batch(FloatTensor(self.normalization(after_state)))
#         next_units = self.transition_model_unit.predict_batch(FloatTensor(self.normalization(after_state)))
        
#         after_state[:, self.index_hp] = next_HPs.cpu().detach().numpy()
#         after_state[:, self.index_units] = next_units.round().cpu().detach().numpy()
        
#         return after_state
        
#     def get_after_states(self, state, actions_self, actions_enemy, fifo_self, fifo_enemy):
#         # Faster combination way
        
#         after_states_self, after_fifo_self = self.combine_sa(state, actions_self, fifo_self, is_enemy = False)
# #         print(len(actions_self), len(after_fifo_self))
# #         after_states_self = self.imply_mineral_by_action(after_states_self, actions_self)
# #         for af, ff in zip(after_states_self, after_fifo_self):
# #             print(af.tolist(), ff)
#         after_states = np.zeros((len(actions_self) * len(actions_enemy), after_states_self.shape[1]))
# #         print(after_states.shape)
# #         idx = 0
#         after_state_actions_self = np.zeros((len(actions_self) * len(actions_enemy), actions_self.shape[1]))
# #         after_state_fifo_self = []
#         for i, a_s_s in enumerate(after_states_self):
#             a_s, _ = self.combine_sa(a_s_s, actions_enemy, fifo_enemy, is_enemy = True)
# #             print(a_s.shape)
#             after_states[i * len(actions_enemy) : (i + 1)* len(actions_enemy)] = a_s
    
#             after_state_actions_self[i * len(actions_enemy) : (i + 1)* len(actions_enemy)] = np.repeat(actions_self[i].reshape((1,-1)), len(actions_enemy), axis = 0).copy()
# #             for _ in range(len(actions_enemy)):
# #                 after_state_fifo_self.append(deepcopy(after_fifo_self[i]))
# #         print(after_states[:, building_types['Pylon']])
# #         print("*********")
# #         print(len(after_states), len(after_fifo_self), len(actions_enemy))
#         after_states[:, self.env.miner_index] += after_states[:, building_types['Pylon']] * 50 + 100
# #             idx += 1
# #         print(idx)
# #         for a_s in after_states:
# #             print(a_s.tolist())
        
#         return after_states, after_fifo_self, after_state_actions_self

#     def combine_sa(self, de_s, actions, fifo, is_enemy):
#         if not is_enemy:
#             building_index = list(range(0, 4))
#         else:
#             building_index = list(range(5, 9))
#         fifo_list = []
#         for _ in range(len(actions)):
#             fifo_list.append(deepcopy(fifo))
#         s = np.repeat(de_s.reshape((1,-1)), len(actions), axis = 0)
#         actions = actions.reshape(-1, 4)
#         for idx_a, action in enumerate(actions):
# #             print(action)
#             for a, num in enumerate(action):
#                 for _ in range(int(num)):
#                     s[idx_a][building_index[a]] += 1
#                     fifo_list[idx_a].append(building_index[a])
#                     if len(fifo_list[idx_a]) > 30:
#                         s[idx_a][building_index[fifo_list[idx_a][0]]] -= 1
#                         del fifo_list[idx_a][0]
#         if not is_enemy:    
#             s[:, self.env.miner_index] -= np.sum(self.env.maker_cost_np * actions, axis = 1)
#         return s, fifo_list
    
#     def imply_mineral_by_action(self, mineral, action):
#         mineral -= np.sum(self.env.maker_cost_np * action)
#         return mineral
        
#     def imply_after_mineral(self, state):
#         state[env.miner_index] += state[building_types['Pylon']] * 50 + 100
#         return state

