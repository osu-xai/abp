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
        
#         self.value_model = DQNModel(self.name + "_eval", self.network_config, use_cuda)
        self.q_model = DQNModel(self.name + "_eval", self.network_config, use_cuda, is_sigmoid = True)
        
    def load_model(self, models_path):
        HP_state_dict = torch.load(models_path + 'transition_model_hp.pt')
        unit_state_dict = torch.load(models_path + 'transition_model_unit.pt')
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
        
#         print(the_HP_keys)
#         print(the_unit_keys)
#         print(the_value_keys)
#         print("*************")
        for i in range(len(the_HP_weight)):
            new_HP_state_dict[the_HP_keys[i]] = the_HP_weight[i]
        for i in range(len(the_HP_weight)):
            new_unit_state_dict[the_unit_keys[i]] = the_unit_weight[i]
#         for i in range(len(the_HP_weight)):
#             new_value_state_dict[the_value_keys[i]] = the_value_weight[i]
            
        self.transition_model_HP.load_weight(new_HP_state_dict)
        self.transition_model_unit.load_weight(new_unit_state_dict)
#         self.value_model.load_weight(new_value_state_dict)
        self.q_model.load_weight(torch.load(models_path + 'q_model_win_prob.pt'))
        
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
        top_k_action_self, top_k_state_self, topk_q_self = self.action_ranking(com_states_self, actions_self)
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
        top_k_action_enemy, top_k_state_enemy, topk_q_enemy = self.action_ranking(com_states_enemy, actions_enemy)
#         print("8---------------------------")
#         print(top_k_action_enemy)
        
        max_value = float("-inf")
        for idx, a_self in tqdm(enumerate(top_k_action_self), desc = "Making decision depth = " + str(depth)):
#             print("9---------------------------")
#             print(a_self)
#             print("10---------------------------")
#             print(top_k_action_enemy)
            action_node_max = Node("dp{}_level{}_action_max".format(dp, depth), top_k_state_self[idx][:-1].tolist(),
                                  parent = state_node, q_value_after_state = topk_q_self[idx].item(), parent_action = a_self.tolist())
            action_node_max.parent.add_child(action_node_max, action = action_node_max.parent_action)
    
            action_node_mins = []
#             print(len(top_k_action_enemy))
#             start_time = time.time()
            next_states = self.get_next_states(state, a_self, top_k_action_enemy)
#             print(time.time() - start_time)
            for tk_a, tk_a_s_e, tk_q_v_e, n_s in zip(top_k_action_enemy, top_k_state_enemy, topk_q_enemy, next_states):
                node_min = Node("dp{}_level{}_action_min".format(dp, depth), self.switch_state_to_enemy(tk_a_s_e)[:-1].tolist(),
                                      parent = action_node_max, q_value_after_state = tk_q_v_e.item(), parent_action = tk_a.tolist())
                action_node_mins.append(node_min)
                node_min.parent.add_child(node_min)
                
            next_reward = self.reward_func_win_prob(state, next_states)
#             print("12---------------------------")
#             print(next_reward)
        
            if depth == self.look_forward_step:
                min_values = self.rollout(next_states)
#                 print(min_values)
#                 input()
                next_state_nodes = []
                for m_v, n_s, a_n_m in zip(min_values, next_states, action_node_mins):
                    node_n_s = Node("dp{}_level{}_state".format(dp, depth), n_s[:-1].tolist(),
                                          parent = a_n_m, best_q_value = m_v.item())
                    a_n_m.best_q_value = m_v.item()
                    a_n_m.best_child = node_n_s
                    next_state_nodes.append(node_n_s)
                    node_n_s.parent.add_child(node_n_s)
#                 min_values[next_reward == 1] = 1
#                 min_values[next_reward == 0] = 0
                
            else:
                next_state_nodes = []
                min_values = FloatTensor(np.zeros(len(next_states)))
                for i, (n_s, n_r) in enumerate(zip(next_states, next_reward)):
#                     if n_r == 1 or n_r == 0:
# #                         min_values[i] = n_r
#                         continue
                    min_values[i], _, next_state_node = self.minimax(n_s, dp = dp + 1, depth = depth + 1, par_node = action_node_mins[i])
                    action_node_mins[i].best_q_value = min_values[i].item()
                    action_node_mins[i].best_child = next_state_node
                    next_state_nodes.append(next_state_node)
                    next_state_node.parent.add_child(next_state_node)
#             print("13---------------------------")
#             print(min_values)
            min_values = min_values.view(-1)
#             print(min_values)
#             min_value = min_values.min(0)[0].item()
            min_value, min_value_idx = min_values.min(0)
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
#         input()
        return best_action, root
    
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

                values[i] = self.q_model.predict_batch(FloatTensor(self.normalization(com_states, is_vf = True)))[1].max(0)[0]
        return values
    
    def action_ranking(self, after_states, action):
        
        ranking_topk = self.ranking_topk
        if ranking_topk >= len(after_states):
              ranking_topk = len(after_states)
                
        with torch.no_grad():
            q_values = self.q_model.predict_batch(FloatTensor(self.normalization(after_states, is_vf = True)))[1]
            q_values = q_values.view(-1)
            topk_q, indices = torch.topk(q_values, ranking_topk)
        
#         print(topk_q)
#         print(indices)
#         input()
        return action[indices.cpu().clone().numpy()], after_states[indices.cpu().clone().numpy()], topk_q
    
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
        after_states[after_states[:, self.mineral_index_self] > 1500] = 1500
        
        after_states[:, self.mineral_index_enemy] += after_states[:, self.pylon_index_enemy] * 75 + 100
        after_states[after_states[:, self.mineral_index_enemy] > 1500] = 1500
        
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
        actions = FloatTensor(actions.copy())
        
#         print(com_state.shape)
#         print(actions.shape)
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