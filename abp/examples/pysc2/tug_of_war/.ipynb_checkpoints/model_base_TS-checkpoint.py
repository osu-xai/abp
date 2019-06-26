import gym
import time
import numpy as np
from absl import flags
import sys, os
import torch

from abp.adaptives import MBTSAdaptive
from abp.utils import clear_summary_path
from abp.explanations import PDX
from tensorboardX import SummaryWriter
from gym.envs.registration import register
from sc2env.environments.tug_of_war_2p import TugOfWar
from sc2env.xai_replay.recorder.recorder import XaiReplayRecorder
from tqdm import tqdm
from copy import deepcopy
from random import randint

np.set_printoptions(precision = 2)
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

def run_task(evaluation_config, network_config, reinforce_config, map_name = None, train_forever = False):
    if (use_cuda):
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("|       USING CUDA       |")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
    else:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("|     NOT USING CUDA     |")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
    flags.FLAGS(sys.argv[:1])
    
    max_episode_steps = 35
    #state = env.reset()
    reward_types = ['player_1_get_damage_2',
                    'player_2_get_damage_1',
                    'player_1_win_1',
                    'player_2_win_2']
    
    choices = [0,1,2,3]
    #pdx_explanation = PDX()
    replay_dimension = evaluation_config.xai_replay_dimension
    env = TugOfWar(reward_types, map_name = map_name, \
        generate_xai_replay = evaluation_config.generate_xai_replay, xai_replay_dimension = replay_dimension)
    
    combine_sa = env.combine_sa
    state_1, state_2 = env.reset()
    
    models_path = "abp/examples/pysc2/tug_of_war/models_mb/"
    agent_1 = MBTSAdaptive(name = "TugOfWar", state_length = len(state_1),
                        network_config = network_config, reinforce_config = reinforce_config,
                          models_path = models_path, env = env)
    agent_1.eval_mode()
    if not reinforce_config.is_random_agent_2:
        agent_2 = SADQAdaptive(name = "TugOfWar",
                            state_length = len(state_2),
                            network_config = network_config,
                            reinforce_config = reinforce_config)
        print("sadq agent 2")
    else:
        print("random agent 2")
        
    test_summaries_path = evaluation_config.summaries_path + "/test"
    clear_summary_path(test_summaries_path)
    test_summary_writer = SummaryWriter(test_summaries_path)
    
    while True:
        if not reinforce_config.is_random_agent_2:
            agent_2.disable_learning()

        total_rewwards_list = []
        # Test Episodes
        print("======================================================================")
        print("===============================Now testing============================")
        print("======================================================================")
                
        for episode in tqdm(range(evaluation_config.test_episodes)):
            state = env.reset()
            total_reward_1 = 0
            done = False
            skiping = True
            steps = 0
            
            while skiping:
                state_1, state_2, done, dp = env.step([], 0)
                
                if dp or done:
                    break
#             input("done stepping to finish prior action")
            while not done and steps < max_episode_steps:
                steps += 1
#                 # Decision point
#                 print('state:')
#                 print(list(env.denormalization(state_1)))
#                 print(list(env.denormalization(state_2)))
                actions_1 = env.get_big_A(env.denormalization(state_1)[env.miner_index])
                actions_2 = env.get_big_A(env.denormalization(state_2)[env.miner_index])
                
                choice_1 = agent_1.predict(env.denormalization(state_1))

                combine_states_2 = combine_sa(state_2, actions_2, 2)
                if not reinforce_config.is_random_agent_2 and not random_enemy:
                    choice_2, _ = agent_2.predict(combine_states_2)
                else:
                    choice_2 = randint(0, len(actions_2) - 1)
                    
                env.step(list(actions_1[choice_1]), 1)
                env.step(list(actions_2[choice_2]), 2)
                
                while skiping:
                    state_1, state_2, done, dp = env.step([], 0)
                    #input(' step wating for done signal')
                    if dp or done:
                        break
#                 input('done stepping after collecting experience')
                current_reward_1 = 0
                reward_1, reward_2 = env.sperate_reward(env.decomposed_rewards)
                for r1 in reward_1:
                    current_reward_1 += r1
                    
                total_reward_1 += current_reward_1

            total_rewwards_list.append(total_reward_1)
            test_summary_writer.add_scalar(tag="Test/Episode Reward", scalar_value=total_reward_1,
                                           global_step=episode + 1)
            test_summary_writer.add_scalar(tag="Test/Steps to choosing Enemies", scalar_value=steps + 1,
                                           global_step=episode + 1)

        tr = sum(total_rewwards_list) / evaluation_config.test_episodes
        print("total reward:")
        print(tr)
        
        f = open("result_model_based.txt", "a+")
        f.write(str(tr) + "\n")
        f.close()
        
def pretty_print(i,data):
#     data = np.stack(np.array(data))
#     print(len(data[i+1][]))
    print("---------------------------------------------- input --------------------------------------------------------------------")
    print("i:\t" + str(i) + "\t\tfriendly nexus: " + str(data[i][0][4]) + "\t\tenemey nexus: " + str(data[i][0][9]))
#     print("i+1:\t" + str(i+1) + "\t\tfriendly nexus: " + str(data[i+1][0][4]) + "\t\tenemey nexus: " + str(data[i+1][0][9]))
    print("\tmarine: " + str(data[i][0][0]) + "\tvikings: " + str(data[i][0][1]) + "\tcolossus: " + str(data[i][0][2]) + "\tpylons: " + str(data[i][0][3]) + "\tE marine: " + str(data[i][0][5]) + "\tE vikings: " + str(data[i][0][6]) + "\tE colossus: " + str(data[i][0][7]) + "\tE pylons: " + str(data[i][0][8]))
    print('on feild:')
    print("\tmarine: " + str(data[i][0][11]) + "\tvikings: " + str(data[i][0][12]) + "\tcolossus: " + str(data[i][0][13]) + "\tE marine: " + str(data[i][0][14]) + "\tE vikings: " + str(data[i][0][15]) + "\tE colossus: " + str(data[i][0][16]))
    print('mineral:' + str(data[i][0][10]))
#     print('reward:' + str(data[i][0][17]))
#     print("\tmarine: " + str(data[i+1][0][0]) + "\tvikings: " + str(data[i+1][0][1]) + "\tcolossus: " + str(data[i+1][0][2]) + "\tpylons: " + str(data[i+1][0][3]) + "\tE marine: " + str(data[i+1][0][5]) + "\tE vikings: " + str(data[i+1][0][6]) + "\tE colossus: " + str(data[i+1][0][7]) + "\tE pylons: " + str(data[i+1][0][8]))
    print("-------------------------------------------------------------------------------------------------------------------------")
    
    print("---------------------------------------------- output ------------------------------------------------------------------------")
    print("i:\t" + str(i) + "\t\tfriendly nexus: " + str(data[i][1][4]) + "\t\tenemey nexus: " + str(data[i][1][9]))
#     print("i+1:\t" + str(i+1) + "\t\tfriendly nexus: " + str(data[i+1][1][4]) + "\t\tenemey nexus: " + str(data[i+1][1][9]))
    print("\tmarine: " + str(data[i][1][0]) + "\tvikings: " + str(data[i][1][1]) + "\tcolossus: " + str(data[i][1][2]) + "\tpylons: " + str(data[i][1][3]) + "\tE marine: " + str(data[i][1][5]) + "\tE vikings: " + str(data[i][1][6]) + "\tE colossus: " + str(data[i][1][7]) + "\tE pylons: " + str(data[i][1][8]))
    print('on feild:')
    print("\tmarine: " + str(data[i][1][11]) + "\tvikings: " + str(data[i][1][12]) + "\tcolossus: " + str(data[i][1][13]) + "\tE marine: " + str(data[i][1][14]) + "\tE vikings: " + str(data[i][1][15]) + "\tE colossus: " + str(data[i][1][16]))
    print('mineral:' + str(data[i][1][10]))
    print('reward:' + str(data[i][1][17]))
#     print("\tmarine: " + str(data[i+1][1][0]) + "\tvikings: " + str(data[i+1][1][1]) + "\tcolossus: " + str(data[i+1][1][2]) + "\tpylons: " + str(data[i+1][1][3]) + "\tE marine: " + str(data[i+1][1][5]) + "\tE vikings: " + str(data[i+1][1][6]) + "\tE colossus: " + str(data[i+1][1][7]) + "\tE pylons: " + str(data[i+1][1][8]))
    print("-------------------------------------------------------------------------------------------------------------------------")
