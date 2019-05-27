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
from sc2env.environments.tug_of_war_bigA import TugOfWar
from sc2env.xai_replay.recorder.recorder import XaiReplayRecorder
from tqdm import tqdm
from copy import deepcopy
from random import randint

np.set_printoptions(precision = 2)
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

def run_task(evaluation_config, network_config, reinforce_config, map_name = None, train_forever = False):
    flags.FLAGS(sys.argv[:1])

    reward_types = ['killEnemyMarine',
                    'killEnemyViking',
                    'killEnemyColossus',
                    'friendlyMarineCasualty_Neg',
                    'friendlyVikingCasualty_Neg',
                    'friendlyColossusCasualty_Neg',
                    'totalIncome',
                    'damageToEnemyBaseHP',
                    'damageToEnemyBaseSheild',
                    'damageToSelfBaseHP_Neg',
                    'damageToSelfBaseSheild_Neg',
                    'win',
                    'loss_Neg']

    max_episode_steps = 35
    #state = env.reset()
    
    #choices = [0,1,2,3,4]
    #pdx_explanation = PDX()
    replay_dimension = evaluation_config.xai_replay_dimension
    env = TugOfWar(reward_types, map_name = map_name, \
        generate_xai_replay = evaluation_config.generate_xai_replay, xai_replay_dimension = replay_dimension)
    combine_sa = env.combine_sa
    state = env.reset()
    agent = SADQAdaptive(name = "TugOfWar",
                        state_length = len(state),
                        network_config = network_config,
                        reinforce_config = reinforce_config)
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
        #for episode in range(1):
            break
            state = env.reset()
            total_reward = 0
            end = False
            skiping = True
            playing = True
            steps = 0

            while skiping:
                state, actions, end, dp = env.step([])
                if dp or end:
                    break
            
            while playing:
                stepRewards = {}
                steps += 1
    #             print("state 1:")
   #             print(state)
                # Decision point
                combine_states = combine_sa(state, actions)
    #             print(combine_states)
                #input('pause')
                if episode / evaluation_config.training_episodes >= 0.75:
                    choice, _ = agent.predict(combine_states, isGreedy = True)
                else:
                    choice, _ = agent.predict(combine_states)
                # assign action
    #             print("action:")
    #             print(list(actions[choice]))
                state, _, _, _ = env.step(list(actions[choice]))
    #            state, _, _, _ = env.step([])
    #             print("state 2:")
    #             print(state)
                while skiping:
                    state, actions, end, dp = env.step([])
                    if dp or end:
                        break

                reward = sum(env.decomposed_rewards[7:11]) - 10
#                 print(np.array(env.decomposed_rewards[:]))
    #             print(total_reward)
                #input('pause')
                agent.reward(reward)

                if steps > max_episode_steps:
                    break
                if end:
                    break

                #print(action)
    #             print("state 3:")
    #             print(state)
    #             print("rewards")
    #             print(np.array(env.decomposed_rewards[7 : 11]))

    #             print()
            #print(steps)
            agent.end_episode(env.end_state)

            test_summary_writer.add_scalar(tag = "Train/Episode Reward", scalar_value = total_reward,
                                           global_step = episode + 1)
            train_summary_writer.add_scalar(tag = "Train/Steps to choosing Enemies", scalar_value = steps + 1,
                                            global_step = episode + 1)

        agent.disable_learning()

        total_rewwards_list = []
            
        # Test Episodes
        print("======================================================================")
        print("===============================Now testing============================")
        print("======================================================================")
        
        collecting_experience = True
        
        all_experiences = []
        for episode in tqdm(range(1000)):

            state = env.reset()
            total_reward = 0
            end = False
            skiping = True
            playing = True
            steps = 0
            previous_state = None
            while skiping:
                state, actions, end, dp = env.step([])
                if dp or end:
                    break
            
            while playing:
                stepRewards = {}
                steps += 1
                
                choice = randint(0, len(actions) - 1)
                #combine_states = combine_sa(state, actions)

                #choice, q_values = agent.predict(combine_states)

                state, _, _, _ = env.step(list(actions[choice]))
                #######
                #experience collecting
                ######
                if collecting_experience:
                    if previous_state is not None:
                        experience = experience_data(env.denormalization(previous_state),
                                                     current_reward,
                                                     env.denormalization(state))
                        #print(experience)
                    
                        all_experiences.append(experience)
                    previous_state = deepcopy(state)
                    
                while skiping:
                    state, actions, end, dp = env.step([])
                    if dp or end:
                        break
    #             print(combine_states)
    #             print(q_values)
    #             print(choice)
    #             print(env.decomposed_rewards[7:11])
                current_reward = sum(env.decomposed_rewards[7:11])
                total_reward += current_reward

#                 input('pause')
                if steps > max_episode_steps:
                    break
                if end:
                    break
                    
            if collecting_experience:
                if previous_state is not None:
                    experience = experience_data(env.denormalization(previous_state),
                                                 current_reward,
                                                 env.denormalization(state))
                    #print(experience)

                    all_experiences.append(experience)
                previous_state = deepcopy(state)
    #         print(total_reward)

            total_rewwards_list.append(total_reward)
            test_summary_writer.add_scalar(tag="Test/Episode Reward", scalar_value=total_reward,
                                           global_step=episode + 1)
            test_summary_writer.add_scalar(tag="Test/Steps to choosing Enemies", scalar_value=steps + 1,
                                           global_step=episode + 1)
        torch.save(all_experiences, 'all_experiences.pt')
        if collecting_experience:
            break
        #print(test.size())
        tr = sum(total_rewwards_list) / evaluation_config.test_episodes
        print("total reward:")
        print(tr)
        f = open("result.txt", "a+")
        f.write(str(tr) + "\n")
        f.close()
        agent.enable_learning()

def experience_data(state, reward, next_state):
    diff = deepcopy(next_state - state)
    action_a, action_b = diff[:4], diff[6:10]
    #print(state, action_a, action_b)
    return (np.hstack((state, action_a, action_b)), np.hstack((reward, next_state)))