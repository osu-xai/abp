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
from sc2env.environments.tug_of_war_2L_self_play import TugOfWar
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
    
    max_episode_steps = 40
    
    replay_dimension = evaluation_config.xai_replay_dimension
    env = TugOfWar(map_name = map_name, \
        generate_xai_replay = evaluation_config.generate_xai_replay, xai_replay_dimension = replay_dimension)
    
    combine_sa = env.combine_sa
    state_1, state_2 = env.reset()
    
    models_path = "abp/examples/pysc2/tug_of_war/models_mb/"
    agent_1 = MBTSAdaptive(name = "TugOfWar", state_length = len(state_1),
                        network_config = network_config, reinforce_config = reinforce_config,
                          models_path = models_path, env = env)
    
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
                actions_1 = env.get_big_A(state_1[env.miner_index], state_1[env.pylon_index])
                actions_2 = env.get_big_A(state_2[env.miner_index], state_2[env.pylon_index], is_train = True)
                
#                 choice_1 = agent_1.predict(env.denormalization(state_1), env.denormalization(state_2)[env.miner_index])
#                 print(state_1)
                actions_1111111 = agent_1.predict(state_1, state_2[env.miner_index])
                
#                 print(actions_1111111)
#                 print(state_1)
#                 input("state_1 checked")
                combine_states_2 = combine_sa(state_2, actions_2)
                if not reinforce_config.is_random_agent_2 and not random_enemy:
                    choice_2, _ = agent_2.predict(env.normalization(combine_states_2))
                else:
                    choice_2 = randint(0, len(actions_2) - 1)
                    
#                 env.step(list(actions_1[choice_1]), 1)
                env.step(list(actions_1111111), 1)
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
        
def pretty_print(state,  text = ""):
    state_list = state.copy().tolist()
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
    print("T:{:^5},{:^5},{:^5},B:{:^5},{:^5},{:^5}".format(
        state[15],state[16],state[17],state[18],state[19],state[20]))
    print("Unit_Enemy")
    print("T:{:^5},{:^5},{:^5},B:{:^5},{:^5},{:^5}".format(
        state[21],state[22],state[23],state[24],state[25],state[26]))
    print("Hit_Point")
    print("S_T:{:^5},S_B{:^5},E_T{:^5},E_B:{:^5}".format(
        state[27],state[28],state[29],state[30]))