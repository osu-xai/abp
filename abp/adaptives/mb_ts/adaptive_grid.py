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
from abp.models.dqn_model_softmax import DQNModel
from abp.utils.search_tree import Node
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

device = torch.device(0)
class MBTSAdaptive(object):
    """Adaptive which uses the Model base Tree search algorithm"""

    def __init__(self, name, state_length, network_config, reinforce_config, 
                 models_path, env, depth, action_ranking, player = 1,
                is_F_all_unit = True, is_F_all_HP = True):
        super(MBTSAdaptive, self).__init__()
        self.name = name
        #self.choices = choices
        self.network_config = network_config
        self.reinforce_config = reinforce_config
        self.explanation = False
        self.state_length = state_length
        self.is_F_all_unit = is_F_all_unit
        self.is_F_all_HP = is_F_all_HP
        
        self.init_network()
        self.load_model(models_path)
        # Global
        self.steps = 0
        self.episode = 0
        
        self.env = env
        self.player = player
        
#         print("%%%%%%%%%%%%%%%%%%%%%%",state_length)
        self.index_hp = LongTensor(range(63, 67))
        self.index_units = LongTensor(range(15, 63))
        self.index_units_top = LongTensor(list(range(15, 27)) + list(range(39, 51)))
        self.index_units_bottom = LongTensor(list(range(27, 39)) + list(range(51, 63)))
        self.index_building_self = LongTensor(range(1, 8))
        self.index_building_enemy = LongTensor(range(8, 15))
        self.pylon_index_self = self.env.pylon_index
        self.pylon_index_enemy = 14
        self.mineral_index_self = self.env.miner_index
        self.mineral_index_enemy = state_length
        self.maker_cost_np = FloatTensor(self.env.maker_cost_np)
        self.index_waves = 67
        
        if network_config.output_shape == 4:
            self.reward_num = 4
            self.combine_decomposed_func = combine_decomposed_func_4
            self.player_1_end_vector = player_1_end_vector_4

        if network_config.output_shape == 8:
            self.reward_num = 8
            self.combine_decomposed_func = combine_decomposed_func_8
            self.player_1_end_vector = player_1_end_vector_8
            

            
        self.norm_vector = FloatTensor([1500,           # Player 1 unspent minerals
                                    30, 30, 10,     # Player 1 top lane building
                                    30, 30, 10,     # Player 1 bottom lane building
                                    3,              # Player 1 pylons
                                    30, 30, 10,     # Player 2 top lane building
                                    30, 30, 10,     # Player 2 bottom lane building
                                    3,              # Player 2 pylons
                                    30, 30, 10,     # Player 1 units top lane grid 1
                                    30, 30, 10,     # Player 1 units top lane grid 2 
                                    30, 30, 10,     # Player 1 units top lane grid 3
                                    30, 30, 10,     # Player 1 units top lane grid 4
                                    30, 30, 10,     # Player 1 units bottom lane grid 1
                                    30, 30, 10,     # Player 1 units bottom lane grid 2 
                                    30, 30, 10,     # Player 1 units bottom lane grid 3
                                    30, 30, 10,     # Player 1 units bottom lane grid 4
                                    30, 30, 10,     # Player 2 units top lane grid 1
                                    30, 30, 10,     # Player 2 units top lane grid 2 
                                    30, 30, 10,     # Player 2 units top lane grid 3
                                    30, 30, 10,     # Player 2 units top lane grid 4
                                    30, 30, 10,     # Player 2 units bottom lane grid 1
                                    30, 30, 10,     # Player 2 units bottom lane grid 2 
                                    30, 30, 10,     # Player 2 units bottom lane grid 3
                                    30, 30, 10,     # Player 2 units bottom lane grid 4
                                    2000, 2000,     # Player 1 Nexus HP (top, bottom)
                                    2000, 2000,     # Player 2 Nexus HP (top, bottom)
                                    40])              # Wave Number)
        
        self.switch_idx = [state_length] + list(range(8, 15)) + list(range(1, 8)) \
                            + list(range(48, 51)) + list(range(45, 48)) + list(range(42, 45)) + list(range(39, 42))\
                            + list(range(60, 63)) + list(range(57, 60)) + list(range(54, 57)) + list(range(51, 54))\
                            + list(range(24, 27)) + list(range(21, 24)) + list(range(18, 21)) + list(range(15, 18))\
                            + list(range(36, 39)) + list(range(33, 36)) + list(range(30, 33)) + list(range(27, 30))\
                            + list(range(65, 67)) + list(range(63, 65)) + [self.index_waves, 0]
        
#         print(self.switch_idx, len(self.switch_idx))
        self.init_input_idx()
        
        self.look_forward_step = depth
        self.ranking_topk = action_ranking#float('inf')
        
        # Generalize it 
        self.eval_mode()
        
    def init_network(self):
        if self.is_F_all_unit:
            network_trans_unit_path = "./tasks/tug_of_war/trans/unit/v2_grid_all/network.yml"
        else:
            network_trans_unit_path = "./tasks/tug_of_war/trans/unit/v2_grid_F1/network.yml"
        
        if self.is_F_all_HP:
            network_trans_hp_path = "./tasks/tug_of_war/trans/health/v2_grid_all/network.yml"
        else:
            network_trans_hp_path = "./tasks/tug_of_war/trans/health/v2_grid_F1/network.yml"
            
        network_trans_unit = NetworkConfig.load_from_yaml(network_trans_unit_path)
        network_trans_hp = NetworkConfig.load_from_yaml(network_trans_hp_path)
        
        self.transition_model_HP = TransModel("TugOfWar2lNexusHealth",network_trans_hp, use_cuda)
        
        self.transition_model_unit = TransModel("TugOfWar2lNexusUnit",network_trans_unit, use_cuda)
        
#         self.value_model = DQNModel(self.name + "_eval", self.network_config, use_cuda)
        self.q_model = DQNModel(self.name + "_eval", self.network_config, use_cuda)
        
    def load_model(self, models_path):
        if self.is_F_all_unit:
            unit_model_name = "trans_units_grid_all.pt"
        else:
            unit_model_name = "trans_units_grid_F1.pt"
            
        if self.is_F_all_HP:
            hp_model_name = "trans_hp_gird_all.pt"
        else:
            hp_model_name = "trans_hp_gird_F1.pt"
        
        
        HP_state_dict = torch.load(models_path + hp_model_name, map_location = device)
        unit_state_dict = torch.load(models_path + unit_model_name, map_location = device)
#         value_state_dict = torch.load(models_path + 'value_model.pt')

#         print(HP_state_dict)
        new_HP_state_dict = OrderedDict()
        new_unit_state_dict = OrderedDict()
        new_value_state_dict = OrderedDict()
        
        the_HP_weight = list(HP_state_dict.values())
        the_unit_weight = list(unit_state_dict.values())
#         the_value_weight = list(value_state_dict.values())
        
        the_HP_keys = list(self.transition_model_HP.model.state_dict().keys())
        the_unit_keys = list(self.transition_model_unit.model.state_dict().keys())
#         the_value_keys = list(self.value_model.model.state_dict().keys())
#         print(HP_state_dict.keys())
#         print(the_HP_keys)
#         print(unit_state_dict.keys())
#         print(the_unit_keys)
        
#         input()
#         print(the_value_keys)
#         print("*************")
        for i in range(len(the_HP_weight)):
            new_HP_state_dict[the_HP_keys[i]] = the_HP_weight[i]
        for i in range(len(the_unit_weight)):
            new_unit_state_dict[the_unit_keys[i]] = the_unit_weight[i]
#         for i in range(len(the_HP_weight)):
#             new_value_state_dict[the_value_keys[i]] = the_value_weight[i]
            
        self.transition_model_HP.load_weight(new_HP_state_dict)
        self.transition_model_unit.load_weight(new_unit_state_dict)
#         self.value_model.load_weight(new_value_state_dict)
        self.q_model.load_weight(torch.load(models_path + 'q_model_decom8_grid.pt', map_location = device))
            
    def reward_func_win_prob(self, state, next_states):
        
        rewards = FloatTensor((torch.ones(next_states.size()[0], dtype=torch.float) * -1).tolist())
#         print(type(rewards))
        min_s, index = next_states[:, self.index_hp].min(1)
#         print(min_s.size(), index.size())
        win = ((min_s.round() <= 0) & (index > 1)).float()
        lose = ((min_s.round() <= 0) & (index <= 1)).float()
#         print(win.size())
        rewards += win * 2
        rewards += lose * 1
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

    def minimax(self, state, dp, depth = 1, par_node = None):
#         print("20---------------------------")
#         print(state)
        state_node = Node("dp{}_level{}_state".format(dp, depth), state[:-1].tolist(), parent = par_node)
#         if par_node is not None:
#             par_node.add_child(state_node)
        
        actions_self = self.env.get_big_A(state[self.mineral_index_self].item(), state[self.pylon_index_self].item())
#         print("3---------------------------")
#         print(actions_self)
        com_states_self = self.combine_sa(state, actions_self, is_enemy = False)
#         print("4---------------------------")
#         print(com_states_self)
        top_k_action_self, top_k_state_self, topk_q_self, com_topk_q_self = self.action_ranking(com_states_self, actions_self)
#         print("5---------------------------")
#         print(top_k_action_self)
    
        actions_enemy = self.env.get_big_A(state[self.mineral_index_enemy].item(), state[self.pylon_index_enemy].item())
#         print("6---------------------------")
#         print(state[self.mineral_index_enemy].item(), actions_enemy)
        # action ranking here
#         print(state)
        com_states_enemy = self.combine_sa(self.switch_state_to_enemy(state), actions_enemy, is_enemy = False)
#         print(state)
#         print("7---------------------------")
#         print(com_states_enemy)
        top_k_action_enemy, top_k_state_enemy, topk_q_enemy, com_topk_q_enemy = self.action_ranking(com_states_enemy, actions_enemy)
#         print("8---------------------------")
#         print(top_k_action_enemy)
        
        max_value = float("-inf")
        for idx, a_self in enumerate(tqdm(top_k_action_self, desc = "Making decision depth={}, dp={}".format(depth, dp))):
#             print("9---------------------------")
#             print(a_self)
#             print("10---------------------------")
#             print(top_k_action_enemy)
            action_node_max = Node("dp{}_level{}_action_max".format(dp, depth), top_k_state_self[idx][:-1].tolist(),
                                  parent = state_node, q_value_after_state = com_topk_q_self[idx].item(), parent_action = a_self.tolist())
            action_node_max.parent.add_child(action_node_max, action = action_node_max.parent_action)
    
            action_node_mins = []
        
#             print(len(top_k_action_enemy))
#             start_time = time.time()
            next_states = self.get_next_states(state, a_self, top_k_action_enemy)
#             print(time.time() - start_time)
            for tk_a, tk_a_s_e, tk_q_v_e, n_s in zip(top_k_action_enemy, top_k_state_enemy, com_topk_q_enemy, next_states):
                node_min = Node("dp{}_level{}_action_min".format(dp, depth), self.switch_state_to_enemy(tk_a_s_e)[:-1].tolist(),
                                      parent = action_node_max, q_value_after_state = tk_q_v_e.item(), parent_action = tk_a.tolist())
                action_node_mins.append(node_min)
                node_min.parent.add_child(node_min)
                
            next_reward = self.reward_func_win_prob(state, next_states)
#             print("12---------------------------")
#             print(next_reward)
        
            if depth == self.look_forward_step:
                decom_min_values, com_min_values = self.rollout(next_states)
#                 print(min_values)
#                 input()
                next_state_nodes = []
                for m_v, n_s, a_n_m in zip(com_min_values, next_states, action_node_mins):
                    node_n_s = Node("dp{}_level{}_state".format(dp, depth), n_s[:-1].tolist(),
                                          parent = a_n_m, best_q_value = m_v.item())
                    a_n_m.best_q_value = m_v.item()
                    a_n_m.best_child = node_n_s
                    next_state_nodes.append(node_n_s)
                    node_n_s.parent.add_child(node_n_s)
#                 com_min_values[next_reward == 1] = 1
#                 com_min_values[next_reward == 0] = 0
                
            else:
                next_state_nodes = []
                com_min_values = FloatTensor(np.zeros(len(next_states)))
                for i, (n_s, n_r) in enumerate(zip(next_states, next_reward)):
#                     if n_r == 1 or n_r == 0:
# #                         min_values[i] = n_r
#                         continue
                    com_min_values[i], _, next_state_node = self.minimax(n_s, dp = dp + 1, depth = depth + 1, par_node = action_node_mins[i])
                    action_node_mins[i].best_q_value = com_min_values[i].item()
                    action_node_mins[i].best_child = next_state_node
                    next_state_nodes.append(next_state_node)
                    next_state_node.parent.add_child(next_state_node)
#             print("13---------------------------")
#             print(min_values)
            com_min_values = com_min_values.view(-1)
#             print(min_values)
#             min_value = min_values.min(0)[0].item()
            min_value, min_value_idx = com_min_values.min(0)
            min_value = min_value.item()
        
            action_node_max.best_q_value = min_value
            action_node_max.best_child = action_node_mins[min_value_idx]
            action_node_max.best_action = action_node_mins[min_value_idx].parent_action
#             print("17---------------------------")
#             print(min_value)
            if max_value < min_value:
                max_value = min_value
                best_action = a_self
                best_max_child = action_node_max
#             print("18---------------------------")
#             print(max_value)
#             print("19---------------------------")
#             print(best_action)
#             input()
        state_node.best_q_value = max_value
        state_node.best_child = best_max_child
        state_node.best_action = best_action
        
        return max_value, best_action, state_node

    def predict(self, state, minerals_enemy, dp = 0):
#         print("1---------------------------")
#         print(state)
        
        state = FloatTensor(np.append(state, minerals_enemy))
#         print("2---------------------------")
#         print(state)
        if self.look_forward_step > 0:
            max_value, best_action, root = self.minimax(state, dp = dp)
        else:
            pass
#         print(state)
#         input()
#         print()
#         print()
#         print()
#         print()
#         print(max_value)
#         pretty_print(state[: -1], text = "original state")
#         pretty_print(self.switch_state_to_enemy(state)[:-1], text = "switch state")
#         input()
        return best_action, root
    
    def normalization(self, state, is_vf = False):
        return state[:, :-1].clone() / self.norm_vector
    
    def denormalization(self, state, is_vf = False):
        return state.clone() * self.norm_vector
    
    def rollout(self, states):
#         return self.value_model.predict_batch(FloatTensor(self.normalization(states, is_vf = True)))

        com_values = FloatTensor(np.zeros(len(states)))
#         print(len(states), self.reward_num)
        values = FloatTensor(np.zeros((len(states), self.reward_num)))
        with torch.no_grad():
            for i, state in enumerate(states):
                actions_self = self.env.get_big_A(state[self.mineral_index_self].item(), state[self.pylon_index_self].item())

                com_states = self.combine_sa(state, actions_self, is_enemy = False)

                decomposed_values = self.q_model.predict_batch(FloatTensor(self.normalization(com_states, is_vf = True)))
                
                com_values[i], idx = self.combine_decomposed_func(decomposed_values).max(0)
                
                values[i] = decomposed_values[idx]
                
        return values, com_values
    
    def action_ranking(self, after_states, action):
#         action = FloatTensor(action)
#         print(len(after_states), len(action))
        ranking_topk = self.ranking_topk
        if ranking_topk >= len(after_states):
              ranking_topk = len(after_states)
#             return action[np.array(range(len(after_states)))], after_states[np.array(range(len(after_states)))]
#         print(ranking_topk)
        with torch.no_grad():
#             print(FloatTensor(self.normalization(after_states, is_vf = True)))
            q_values = self.q_model.predict_batch(FloatTensor(self.normalization(after_states, is_vf = True)))
#             print(q_values)
            com_q_values = self.combine_decomposed_func(q_values)
            
            com_q_values = com_q_values.view(-1)
            com_topk_q, indices = torch.topk(com_q_values, ranking_topk)
            topk_q = q_values[indices]
#         print(topk_q)
#         print(indices)
#         input()
        return action[indices.cpu().clone().numpy()], after_states[indices.cpu().clone().numpy()], topk_q, com_topk_q
    
    def switch_state_to_enemy(self, states):
        return states[self.switch_idx].clone()
    
    def get_next_states(self, state, a_self, top_k_action_enemy):
#         print("****************************")
        after_states = self.get_after_states(state, a_self, top_k_action_enemy)
        next_states = after_states.clone()
        next_states[:, self.index_waves] += 1
        
#         pretty_print(next_states[5][: -1])
        len_n_s = next_states.size()[0]
        input_units_top, input_units_bottom, input_hp_1_idx, input_hp_2_idx, input_hp_3_idx, input_hp_4_idx = self.separate_state(self.normalization(next_states))
        
        
        with torch.no_grad():
            if self.is_F_all_unit:
                next_units = self.transition_model_unit.predict_batch(FloatTensor(self.normalization(next_states))).round()
            else:
                input_units = torch.cat((input_units_top, input_units_bottom))
                next_units = self.transition_model_unit.predict_batch(input_units).round()

            if self.is_F_all_HP:
                next_HPs = self.transition_model_HP.predict_batch(FloatTensor(self.normalization(next_states))).round()
                
            else:
                input_hps = torch.cat((input_hp_1_idx, input_hp_2_idx, input_hp_3_idx, input_hp_4_idx))
                next_HPs = self.transition_model_HP.predict_batch(input_hps).round()
                
            next_units[next_units < 0] = 0
            next_HPs[next_HPs < 0] = 0
            next_HPs[next_HPs > 2000] = 2000
                     
            if self.is_F_all_unit:
                next_states[:, self.index_units] = next_units
            else:
                next_states[:, self.index_units_top] = next_units[: len_n_s]
                next_states[:, self.index_units_bottom] = next_units[len_n_s : ]            
            
            if self.is_F_all_HP:
                next_states[:, self.index_hp] = next_HPs
            else:
                for i in range(4):
                    next_states[:, self.index_hp[i]] = next_HPs[i * len_n_s : (i + 1) * len_n_s, 0]
        
#         print(next_states[5].size())
#         pretty_print(next_states[5][: -1])
        
#         input()
        return next_states

    def get_after_states(self, state, a_self, top_k_action_enemy):
        # Faster combination way
        after_states_self = self.combine_sa(state, np.array(a_self).reshape(-1, 7), False)
        after_states = self.combine_sa(after_states_self, top_k_action_enemy, True)
        
        after_states[:, self.mineral_index_self] += after_states[:, self.pylon_index_self] * 75 + 100
        after_states[after_states[:, self.mineral_index_self] > 1500] = 1500
        
        after_states[:, self.mineral_index_enemy] += after_states[:, self.pylon_index_enemy] * 75 + 100
        after_states[after_states[:, self.mineral_index_enemy] > 1500] = 1500
        
        return after_states
    
    def init_input_idx(self):
        
        self_buildings_top = list(range(1, 4))
        self_buildings_bottom = list(range(4, 7))
        
        enemy_buildings_top = list(range(8, 11))
        enemy_buildings_bottom = list(range(11, 14))
        
        self_units_top = list(range(15, 27))
        self_units_bottom = list(range(27, 39))
        
        self_units_top_reversed = list(range(24, 27)) + list(range(21, 24)) + list(range(18, 21)) + list(range(15, 18))
        
        self_units_bottom_reversed = list(range(36, 39)) + list(range(33, 36)) + list(range(30, 33)) + list(range(27, 30))
        
        enemy_units_top = list(range(39, 51))
        enemy_units_bottom = list(range(51, 63))
        
        enemy_units_top_reversed = list(range(48, 51)) + list(range(45, 48)) + list(range(42, 45)) + list(range(39, 42))
        enemy_units_bottom_reversed = list(range(60, 63)) + list(range(57, 60)) + list(range(54, 57)) + list(range(51, 54))
        
        self_hp_top = [63]
        self_hp_bottom = [64]
        
        enemy_hp_top = [65]
        enemy_hp_bottom = [66]
        
        self.input_units_top_idx = self_buildings_top + enemy_buildings_top + self_units_top + enemy_units_top

        self.input_units_bottom_idx = self_buildings_bottom + enemy_buildings_bottom + self_units_bottom + enemy_units_bottom
        
        self.input_hp_1_idx = self_buildings_top + enemy_buildings_top + self_units_top + enemy_units_top + self_hp_top

        self.input_hp_2_idx = self_buildings_bottom + enemy_buildings_bottom + self_units_bottom + enemy_units_bottom + self_hp_bottom

        self.input_hp_3_idx = enemy_buildings_top + self_buildings_top + enemy_units_top_reversed + self_units_top_reversed + enemy_hp_top

        self.input_hp_4_idx = enemy_buildings_bottom + self_buildings_bottom + enemy_units_bottom_reversed + self_units_bottom_reversed + enemy_hp_bottom
            
#         print(self.input_1_idx)
#         print(self.input_2_idx)
#         print(self.input_3_idx)
#         print(self.input_4_idx)
#         input()
    
    def separate_state(self, state):
        input_units_top = state[:, self.input_units_top_idx].clone()
        input_units_bottom = state[:, self.input_units_bottom_idx].clone()
        input_hp_1_idx = state[:, self.input_hp_1_idx].clone()
        input_hp_2_idx = state[:, self.input_hp_2_idx].clone()
        input_hp_3_idx = state[:, self.input_hp_3_idx].clone()
        input_hp_4_idx = state[:, self.input_hp_4_idx].clone()
        
        return input_units_top, input_units_bottom, input_hp_1_idx, input_hp_2_idx, input_hp_3_idx, input_hp_4_idx
        
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
        actions = FloatTensor(actions.copy())
        
        com_state[:,building_index[:-1]] += actions[:, : -1]
        
        com_state[:, mineral_index] -= torch.sum(self.maker_cost_np * actions[:, :-1], dim = 1)
        
        index_has_pylon = actions[:, -1] > 0
        while sum(index_has_pylon) != 0:
            num_of_pylon = com_state[index_has_pylon, pylon_index]
            com_state[index_has_pylon, pylon_index] += 1
            com_state[index_has_pylon, mineral_index] -= (self.env.pylon_cost + num_of_pylon * 100)
            
            actions[index_has_pylon, -1] -= 1
            index_has_pylon = actions[:, -1] > 0
        
        assert torch.sum(com_state[:, mineral_index] >= 0) == len(com_state), print(com_state)
        return com_state
  
    
def combine_decomposed_func_4(q_values):
#     print(q_values)
#     print(q_values[:, :1].size())
    q_values = torch.sum(q_values[:, 2:], dim = 1)
#     print(q_values)
#     input("combine")
    return q_values

def combine_decomposed_func_8(q_values):
#     print(q_values)
#     print(q_values[:, :1].size())
    q_values = torch.sum(q_values[:, [2, 3, 6, 7]], dim = 1)
#     print(q_values)
#     input("combine")
    return q_values
            
def player_1_end_vector_4(state_1_T_hp, state_1_B_hp, state_2_T_hp, state_2_B_hp, is_done = False):
    hp_vector = np.array([state_1_T_hp, state_1_B_hp, state_2_T_hp, state_2_B_hp])
    min_value, idx = np.min(hp_vector), np.argmin(hp_vector)
    
    if min_value == 2000:
        reward_vector = [0.25] * 4
    else:
        reward_vector = [0] * 4
        reward_vector[idx] = 1
#     print(hp_vector)
#     print(reward_vector)
#     input("reward_vector")
    return reward_vector

def player_1_end_vector_8(state_1_T_hp, state_1_B_hp, state_2_T_hp, state_2_B_hp, is_done = False):
    
    hp_vector = np.array([state_1_T_hp, state_1_B_hp, state_2_T_hp, state_2_B_hp])
    min_value, idx = np.min(hp_vector), np.argmin(hp_vector)
    
    if not is_done:
        if min_value == 2000:
            reward_vector = [0] * 4 + [0.25] * 4
        else:
            reward_vector = [0] * 8
            reward_vector[idx + 4] = 1
    else:
        reward_vector = [0] * 8
        reward_vector[idx] = 1
        
#     print(hp_vector)
#     print(reward_vector)
#     input("reward_vector")

    return reward_vector

def pretty_print(state,  text = ""):
    state_list = state.tolist()
    state = []
    for s in state_list:
        state.append(str(s))
    print("===========================================")
    print(text)
    print("Wave:\t" + state[-1])
    print("Minerals:\t" + state[0])
    print("Building_Self")
    print("T:{:^5},{:^5},{:^5},B:{:^5},{:^5},{:^5},P:{:^5}".format(
        state[1],state[2],state[3],state[4],state[5],state[6],state[7]))
    print("Building_Enemy")
    print("T:{:^5},{:^5},{:^5},B:{:^5},{:^5},{:^5},P:{:^5}".format(
        state[8],state[9],state[10],state[11],state[12],state[13],state[14]))
    
    print("Unit_Self")
    print("     M  ,  B  ,  I ")
    print("T1:{:^5},{:^5},{:^5}".format(
        state[15],state[16],state[17]))

    print("T2:{:^5},{:^5},{:^5}".format(
        state[18],state[19],state[20]))

    print("T3:{:^5},{:^5},{:^5}".format(
        state[21],state[22],state[23]))

    print("T4:{:^5},{:^5},{:^5}".format(
        state[24],state[25],state[26]))

    print("B1:{:^5},{:^5},{:^5}".format(
        state[27],state[28],state[29]))

    print("B2:{:^5},{:^5},{:^5}".format(
        state[30],state[31],state[32]))

    print("B3:{:^5},{:^5},{:^5}".format(
        state[33],state[34],state[35]))

    print("B4:{:^5},{:^5},{:^5}".format(
        state[36],state[37],state[38]))

    print("Unit_Enemy")
    print("     M  ,  B  ,  I ")
    print("T1:{:^5},{:^5},{:^5}".format(
        state[39],state[40],state[41]))

    print("T2:{:^5},{:^5},{:^5}".format(
        state[42],state[43],state[44]))

    print("T3:{:^5},{:^5},{:^5}".format(
        state[45],state[46],state[47]))

    print("T4:{:^5},{:^5},{:^5}".format(
        state[48],state[49],state[50]))

    print("B1:{:^5},{:^5},{:^5}".format(
        state[51],state[52],state[53]))

    print("B2:{:^5},{:^5},{:^5}".format(
        state[54],state[55],state[56]))

    print("B3:{:^5},{:^5},{:^5}".format(
        state[57],state[58],state[59]))

    print("B4:{:^5},{:^5},{:^5}".format(
        state[60],state[61],state[62]))

    print("Hit_Point")
    print("S_T:{:^5},S_B{:^5},E_T{:^5},E_B:{:^5}".format(
        state[63],state[64],state[65],state[66]))
    
def pretty_print_input(state,  text = ""):
    state_list = state.tolist()
    state = []
    for s in state_list:
        state.append(str(s))
    print("===========================================")
    print(text)
    print("Building_Self")
    print("{:^5},{:^5},{:^5}".format(
        state[0],state[1],state[2]))
    print("Building_Enemy")
    print("{:^5},{:^5},{:^5}".format(
        state[3],state[4],state[5]))
    
    print("Unit_Self")
    print("     M  ,  B  ,  I ")
    print("{:^5},{:^5},{:^5}".format(
        state[6],state[7],state[8]))

    print("{:^5},{:^5},{:^5}".format(
        state[9],state[10],state[11]))

    print("{:^5},{:^5},{:^5}".format(
        state[12],state[13],state[14]))
    
    print("{:^5},{:^5},{:^5}".format(
        state[15],state[16],state[17]))
    print("Unit_Enemy")
    print("     M  ,  B  ,  I ")
    print("{:^5},{:^5},{:^5}".format(
        state[18],state[19],state[20]))

    print("{:^5},{:^5},{:^5}".format(
        state[21],state[22],state[23]))

    print("{:^5},{:^5},{:^5}".format(
        state[24],state[25],state[26]))

    print("{:^5},{:^5},{:^5}".format(
        state[27],state[28],state[29]))

    print("Hit_Point", state[-1])