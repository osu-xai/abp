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
    
    choices = [0,1,2,3]
    #pdx_explanation = PDX()
    replay_dimension = evaluation_config.xai_replay_dimension
    env = TugOfWar(reward_types, map_name = map_name, \
        generate_xai_replay = evaluation_config.generate_xai_replay, xai_replay_dimension = replay_dimension)
    combine_sa = env.combine_sa
    state_1, state_2 = env.reset()
    
    if not reinforce_config.is_random_agent_1:
        agent_1 = SADQAdaptive(name = "TugOfWar",
                            state_length = len(state_1) + len(choices),
                            network_config = network_config,
                            reinforce_config = reinforce_config)
        print("sadq agent 1")
    else:
        print("random agent 1")
    if not reinforce_config.is_random_agent_2:
        agent_2 = SADQAdaptive(name = "TugOfWar",
                            state_length = len(state_2) + len(choices),
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
    
    enemy_update = 20
    
    round_num = 0
    while True:
        if round_num % enemy_update == 0 and not reinforce_config.is_random_agent_2:
            print("replace enemy agent's weight with self agent")
            agent_2.load_model(agent_1.eval_model)
        if not reinforce_config.is_random_agent_2:
            agent_2.disable_learning()
        round_num += 1
        
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
#             print(list(env.denormalization(state_1)))
#             print(list(env.denormalization(state_2)))
            while skiping:
                state_1, state_2, done, dp = env.step([], 0)
                if dp or done:
                    break
            
            while not done and steps < max_episode_steps:
                steps += 1
                # Decision point
#                 print('state:')
#                 print(list(env.denormalization(state_1)))
#                 print(list(env.denormalization(state_2)))
                actions_1 = env.get_big_A(env.denormalization(state_1)[env.miner_index])
                actions_2 = env.get_big_A(env.denormalization(state_2)[env.miner_index])
                
                if not reinforce_config.is_random_agent_1:
                    combine_states_1 = combine_sa(state_1, actions_1)
                    choice_1, _ = agent_1.predict(combine_states_1)
#                     print(combine_states_1)
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
# #                 assign action
#                 print("choice:")
#                 print(actions_1[choice_1])
#                 print(actions_2[choice_2])
#                 input('pause')
                env.step(list(actions_1[choice_1]), 1)
                env.step(list(actions_2[choice_2]), 2)
#                 env.step((0,0,0,1), 1)
#                 env.step((0,0,0,1), 2)
                while skiping:
                    state_1, state_2, done, dp = env.step([], 0)
#                     input('time_step')
                    if dp or done:
                        break

                reward_1, reward_2 = env.sperate_reward(env.decomposed_rewards)
#                 print('reward:')
#                 print(reward_1)
#                 print(reward_2)
                
                for r1, r2 in zip(reward_1, reward_2):
                    if not reinforce_config.is_random_agent_1:
                        agent_1.reward(r1)
#                     if not reinforce_config.is_random_agent_2:
#                         agent_2.reward(r2)
            
            if not reinforce_config.is_random_agent_1:
                agent_1.end_episode(np.hstack((env.end_state_1, np.zeros(4))))
#             if not reinforce_config.is_random_agent_2:
#                 agent_2.end_episode(np.hstack((env.end_state_2, np.zeros(4))))

            test_summary_writer.add_scalar(tag = "Train/Episode Reward", scalar_value = total_reward,
                                           global_step = episode + 1)
            train_summary_writer.add_scalar(tag = "Train/Steps to choosing Enemies", scalar_value = steps + 1,
                                            global_step = episode + 1)
        
        if not reinforce_config.is_random_agent_1:
            agent_1.disable_learning()
            
        if not reinforce_config.is_random_agent_2:
            agent_2.disable_learning()

        total_rewwards_list = []
            
        # Test Episodes
        print("======================================================================")
        print("===============================Now testing============================")
        print("======================================================================")
        
        collecting_experience = False
        
        all_experiences = []
        for episode in tqdm(range(evaluation_config.test_episodes)):
            state = env.reset()
            total_reward_1 = 0
            done = False
            skiping = True
            steps = 0
            previous_state = None
            
            while skiping:
                state_1, state_2, done, dp = env.step([], 0)
                if dp or done:
                    break
            
            while not done and steps < max_episode_steps:
                steps += 1
#                 # Decision point
#                 print('state:')
#                 print(list(env.denormalization(state_1)))
#                 print(list(env.denormalization(state_2)))
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
                    
                env.step(list(actions_1[choice_1]), 1)
                env.step(list(actions_2[choice_2]), 2)
                #######
                #experience collecting
                ######
                if collecting_experience:
                    if previous_state is not None:
                        experience = [np.hstack((env.denormalization(previous_state), 
                                                actions_1[choice_1],
                                                actions_2[choice_2],
                                                np.array([current_reward_1]))),
                                     env.denormalization(state_1)]
                        print(experience)
                        all_experiences.append(experience)
                        
                    previous_state = deepcopy(state_1)
#                 input("123")
                while skiping:
                    state_1, state_2, done, dp = env.step([], 0)
#                     input('time_step')
                    if dp or done:
                        break

                current_reward_1 = 0
                reward_1, reward_2 = env.sperate_reward(env.decomposed_rewards)
                for r1 in reward_1:
                    current_reward_1 += r1
                    
                total_reward_1 += current_reward_1
                
#             if collecting_experience:
#                 if previous_state is not None:
#                     experience = experience_data(env.denormalization(previous_state),
#                                                  current_reward_1,
#                                                  env.denormalization(state))
#                     all_experiences.append(experience)
                    

            total_rewwards_list.append(total_reward_1)
            test_summary_writer.add_scalar(tag="Test/Episode Reward", scalar_value=total_reward_1,
                                           global_step=episode + 1)
            test_summary_writer.add_scalar(tag="Test/Steps to choosing Enemies", scalar_value=steps + 1,
                                           global_step=episode + 1)
        if collecting_experience:        
            break
        #print(test.size())
        tr = sum(total_rewwards_list) / evaluation_config.test_episodes
        print("total reward:")
        print(tr)
        f = open("result_self_play.txt", "a+")
        f.write(str(tr) + "\n")
        f.close()
        if not reinforce_config.is_random_agent_1:
            agent_1.enable_learning()
            
#         if not reinforce_config.is_random_agent_2:
#             agent_2.enable_learning()
        
    if collecting_experience:
        torch.save(all_experiences, 'abp/examples/pysc2/tug_of_war/all_experiences_2.pt')