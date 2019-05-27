import gym
import time
import numpy as np
from absl import flags
import sys, os
import torch

from abp import SADQAdaptive
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
    flags.FLAGS(sys.argv[:1])
    
    # at the end of the reward type name:
    # 1 means for player 1 is positive, for player 2 is negative
    # 2 means for player 2 is positive, for player 1 is negative
    reward_types = ['player_1_get_damage_2',
                    'player_2_get_damage_1',
                    'player_1_win_1',
                    'player_2_win_2']

    max_episode_steps = 35
    #state = env.reset()
    
    #choices = [0,1,2,3,4]
    #pdx_explanation = PDX()
    replay_dimension = evaluation_config.xai_replay_dimension
    env = TugOfWar(reward_types, map_name = map_name, \
        generate_xai_replay = evaluation_config.generate_xai_replay, xai_replay_dimension = replay_dimension)
    combine_sa = env.combine_sa
    state_1, state_2 = env.reset()
    
    if not reinforce_config.is_random_agent_1:
        agent_1 = SADQAdaptive(name = "TugOfWar",
                            state_length = len(state_1),
                            network_config = network_config,
                            reinforce_config = reinforce_config)
        print("sadq agent 1")
    else:
        print("random agent 1")
    if not reinforce_config.is_random_agent_2:
        agent_2 = SADQAdaptive(name = "TugOfWar",
                            state_length = len(state_2),
                            network_config = network_config,
                            reinforce_config = reinforce_config)
        print("sadq agent 2")
    else:
        print("random agent 2")
        
    training_summaries_path = evaluation_config.summaries_path + "/train"
    clear_summary_path(training_summaries_path)
    train_summary_writer = SummaryWriter(training_summaries_path)

    test_summaries_path = evaluation_config.summaries_path + "/test"
    clear_summary_path(test_summaries_path)
    test_summary_writer = SummaryWriter(test_summaries_path)
    
    while True:
        print("=======================================================================")
        print("===============================Now training============================")
        print("=======================================================================")
        print("Now training.")
        for episode in tqdm(range(evaluation_config.training_episodes)):
#         for episode in range(1):
#             break
            state_1, state_2 = env.reset()
            total_reward = 0
            skiping = True
            done = False
            steps = 0
            print(list(env.denormalization(state_1)))
            print(list(env.denormalization(state_2)))
            while skiping:
                state_1, state_2, done, dp = env.step([], 0)
                if dp or done:
                    break
            
            while not done and steps < max_episode_steps:
                steps += 1
                # Decision point
                print('state:')
                print(list(env.denormalization(state_1)))
                print(list(env.denormalization(state_2)))
                actions_1 = env.get_big_A(env.denormalization(state_1)[env.miner_index])
                actions_2 = env.get_big_A(env.denormalization(state_2)[env.miner_index])
                
                if not reinforce_config.is_random_agent_1:
                    combine_states_1 = combine_sa(state_1, actions_1)
                    choice_1, _ = agent_1.predict(combine_states_1)
                else:
                    choice_1 = randint(0, len(actions_1) - 1)
                    
                if not reinforce_config.is_random_agent_2:
                    combine_states_2 = combine_sa(state_2, actions_2)
                    choice_2, _ = agent_2.predict(combine_states_2)
                else:
                    choice_2 = randint(0, len(actions_2) - 1)
#                 print("action list:")
#                 print(actions_1)
#                 print(actions_2)
#                 # assign action
#                 print("choice:")
#                 print(actions_1[choice_1])
#                 print(actions_2[choice_2])
#                 input('pause')
#                 env.step(list(actions_1[choice_1]), 1)
#                 env.step(list(actions_2[choice_2]), 2)
                env.step((0,0,1,1), 1)
                env.step((0,1,0,1), 2)
                while skiping:
                    state_1, state_2, done, dp = env.step([], 0)
                    input('time_step')
                    if dp or done:
                        break

                reward_1, reward_2 = env.sperate_reward(env.decomposed_rewards)
                print('reward:')
                print(reward_1)
                print(reward_2)
                
                for r1, r2 in zip(reward_1, reward_2):
                    if not reinforce_config.is_random_agent_1:
                        agent_1.reward(r1)
                    if not reinforce_config.is_random_agent_2:
                        agent_2.reward(r2)
            
            if not reinforce_config.is_random_agent_1:
                agent_1.end_episode(env.end_state_1)
            if not reinforce_config.is_random_agent_2:
                agent_2.end_episode(env.end_state_2)

            test_summary_writer.add_scalar(tag = "Train/Episode Reward", scalar_value = total_reward,
                                           global_step = episode + 1)
            train_summary_writer.add_scalar(tag = "Train/Steps to choosing Enemies", scalar_value = steps + 1,
                                            global_step = episode + 1)

#         agent.disable_learning()

#         total_rewwards_list = []
            
#         # Test Episodes
#         print("======================================================================")
#         print("===============================Now testing============================")
#         print("======================================================================")
        
#         collecting_experience = True
        
#         all_experiences = []
#         for episode in tqdm(range(1000)):

#             state = env.reset()
#             total_reward = 0
#             end = False
#             skiping = True
#             playing = True
#             steps = 0
#             previous_state = None
#             while skiping:
#                 state, actions, end, dp = env.step([], 0)
#                 if dp or end:
#                     break
            
#             while playing:
#                 stepRewards = {}
#                 steps += 1
                
#                 choice = randint(0, len(actions) - 1)
#                 #combine_states = combine_sa(state, actions)

#                 #choice, q_values = agent.predict(combine_states)

#                 state, _, _, _ = env.step(list(actions[choice]))
#                 #######
#                 #experience collecting
#                 ######
#                 if collecting_experience:
#                     if previous_state is not None:
#                         experience = experience_data(env.denormalization(previous_state),
#                                                      current_reward,
#                                                      env.denormalization(state))
#                         #print(experience)
                    
#                         all_experiences.append(experience)
#                     previous_state = deepcopy(state)
                    
#                 while skiping:
#                     state, actions, end, dp = env.step([])
#                     if dp or end:
#                         break
#     #             print(combine_states)
#     #             print(q_values)
#     #             print(choice)
#     #             print(env.decomposed_rewards[7:11])
#                 current_reward = sum(env.decomposed_rewards[7:11])
#                 total_reward += current_reward

# #                 input('pause')
#                 if steps > max_episode_steps:
#                     break
#                 if end:
#                     break
                    
#             if collecting_experience:
#                 if previous_state is not None:
#                     experience = experience_data(env.denormalization(previous_state),
#                                                  current_reward,
#                                                  env.denormalization(state))
#                     #print(experience)

#                     all_experiences.append(experience)
#                 previous_state = deepcopy(state)
#     #         print(total_reward)

#             total_rewwards_list.append(total_reward)
#             test_summary_writer.add_scalar(tag="Test/Episode Reward", scalar_value=total_reward,
#                                            global_step=episode + 1)
#             test_summary_writer.add_scalar(tag="Test/Steps to choosing Enemies", scalar_value=steps + 1,
#                                            global_step=episode + 1)
#         torch.save(all_experiences, 'all_experiences.pt')
#         if collecting_experience:
#             break
#         #print(test.size())
#         tr = sum(total_rewwards_list) / evaluation_config.test_episodes
#         print("total reward:")
#         print(tr)
#         f = open("result.txt", "a+")
#         f.write(str(tr) + "\n")
#         f.close()
#         agent.enable_learning()

def experience_data(state, reward, next_state):
    diff = deepcopy(next_state - state)
    action_a, action_b = diff[:4], diff[6:10]
    #print(state, action_a, action_b)
    return (np.hstack((state, action_a, action_b)), np.hstack((reward, next_state)))