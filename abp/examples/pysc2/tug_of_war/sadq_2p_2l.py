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

    max_episode_steps = 10
    
    #pdx_explanation = PDX()
    replay_dimension = evaluation_config.xai_replay_dimension
    env = TugOfWar(map_name = map_name, \
        generate_xai_replay = evaluation_config.generate_xai_replay, xai_replay_dimension = replay_dimension)
    reward_types = env.reward_types
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
    
    random_enemy = True
    
    round_num = 0
    
    privous_result = []
    update_wins_waves = 10
    
    all_experiences = []
    
    if not reinforce_config.is_random_agent_2:
        agent_2.load_model(agent_1.eval_model)
        
    while True:
        if len(privous_result) >= update_wins_waves and \
        sum(privous_result) / update_wins_waves > 13000 and \
        not reinforce_config.is_random_agent_2:
            print("replace enemy agent's weight with self agent")
            random_enemy = False
            f = open("result_self_play.txt", "a+")
            f.write("Update agent\n")
            f.close()
            agent_2.load_model(agent_1.eval_model)
            agent_1.steps = reinforce_config.epsilon_timesteps / 2
            agent_1.best_reward_mean = 0
            agent_1.save(force = True, appendix = "update_" + str(round_num))
            
        if not reinforce_config.is_random_agent_2:
            agent_2.disable_learning()
        round_num += 1
    
        
        print("=======================================================================")
        print("===============================Now training============================")
        print("=======================================================================")
        print("Now training.")
        if random_enemy:
            print("enemy is random")
        for episode in tqdm(range(evaluation_config.training_episodes)):
#         for episode in range(1):
#             break
            if reinforce_config.collecting_experience:
                break
            state_1, state_2 = env.reset()
            total_reward = 0
            skiping = True
            done = False
            steps = 0
#             print(list(state_1))
#             print(list(state_2))
            while skiping:
                state_1, state_2, done, dp = env.step([], 0)
                if dp or done:
                    break
            while not done and steps < max_episode_steps:
                steps += 1
                # Decision point
#                 print('state:')
                print("=======================================================================")
                pretty_print(state_1, text = "state 1")
                pretty_print(state_2, text = "state 2")
                actions_1 = env.get_big_A(state_1[env.miner_index], state_1[env.pylon_index])
                actions_2 = env.get_big_A(state_2[env.miner_index], state_1[env.pylon_index])
                
                if not reinforce_config.is_random_agent_1:
                    combine_states_1 = combine_sa(state_1, actions_1)
                    choice_1, _ = agent_1.predict(combine_states_1)
#                     print(combine_states_1)
                else:
                    choice_1 = randint(0, len(actions_1) - 1)
                    
                if not reinforce_config.is_random_agent_2 and not random_enemy:
                    combine_states_2 = combine_sa(state_2, actions_2)
                    choice_2, _ = agent_2.predict(combine_states_2)
                else:
                    choice_2 = randint(0, len(actions_2) - 1)
#                 print("action list:")
#                 print(actions_1)
#                 print(actions_2)
#                 assign action
#                 print("choice:")
#                 print(actions_1[choice_1])
#                 print(actions_2[choice_2])
#                 print("after state:")
#                 print(combine_states_1[choice_1].tolist())
#                 print(combine_states_2[choice_2].tolist())
#                 input('pause')
                env.step(list(actions_1[choice_1]), 1)
                env.step(list(actions_2[choice_2]), 2)
#                 env.step((0,1,0,0,0,0,0), 1)
#                 env.step((0,1,0,0,0,0,0), 2)
                while skiping:
                    state_1, state_2, done, dp = env.step([], 0)
#                     input('time_step')
                    if dp or done:
                        break
#                 print(steps)
                if steps == max_episode_steps and not done:
                    if sum(state_1[27:29]) > sum(state_1[29:31]):
                        env.decomposed_rewards[4] = 10000
                    else:
                        env.decomposed_rewards[5] = 10000
                reward_1, reward_2 = env.sperate_reward(env.decomposed_rewards)
                print('reward:')
                print(reward_1)
                print(reward_2)
                input()

            
                for r1, r2 in zip(reward_1, reward_2):
                    if not reinforce_config.is_random_agent_1:
                        agent_1.reward(r1)
#                     if not reinforce_config.is_random_agent_2:
#                         agent_2.reward(r2)
            
            if not reinforce_config.is_random_agent_1:
                agent_1.end_episode(env.end_state_1)
#             if not reinforce_config.is_random_agent_2:
#                 agent_2.end_episode(np.hstack((env.end_state_2, np.zeros(4))))

            test_summary_writer.add_scalar(tag = "Train/Episode Reward", scalar_value = total_reward,
                                           global_step = episode + 1)
            train_summary_writer.add_scalar(tag = "Train/Steps to choosing Enemies", scalar_value = steps + 1,
                                            global_step = episode + 1)
        
        if not reinforce_config.is_random_agent_1:
            agent_1.disable_learning(is_save = not reinforce_config.collecting_experience)
            
        if not reinforce_config.is_random_agent_2:
            agent_2.disable_learning()

        total_rewwards_list = []
            
        # Test Episodes
        print("======================================================================")
        print("===============================Now testing============================")
        print("======================================================================")
                
        
        for episode in tqdm(range(evaluation_config.test_episodes)):
            env.reset()
            total_reward_1 = 0
            done = False
            skiping = True
            steps = 0
            previous_state_1 = None
            previous_state_2 = None
            previous_action_1 = None
            previous_action_2 = None 
#             print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Starting episode%%%%%%%%%%%%%%%%%%%%%%%%%")
            
            while skiping:
#                 start_time = time.time()
                state_1, state_2, done, dp = env.step([], 0)
                if dp or done:
#                     print(time.time() - start_time)
                    break
#             input("done stepping to finish prior action")
            while not done and steps < max_episode_steps:
                steps += 1
#                 # Decision point
#                 print('state:')
#                 print(list(state_1))
#                 print(list(state_2))
#                 print("Get actions time:")
#                 start_time = time.time()
                actions_1 = env.get_big_A(state_1[env.miner_index], state_1[env.pylon_index])
                actions_2 = env.get_big_A(state_2[env.miner_index], state_1[env.pylon_index])
#                 print(time.time() - start_time)

                combine_states_1 = combine_sa(state_1, actions_1)
                if not reinforce_config.is_random_agent_1:
                    start_time = time.time()

                    choice_1, _ = agent_1.predict(combine_states_1)
                    print(time.time() - start_time)
                else:
                    choice_1 = randint(0, len(actions_1) - 1)
                
                combine_states_2 = combine_sa(state_2, actions_2)
                if not reinforce_config.is_random_agent_2 and not random_enemy:
                    choice_2, _ = agent_2.predict(combine_states_2)
                else:
                    choice_2 = randint(0, len(actions_2) - 1)
                    

#                 input('stepped with command 2')
                #######
                #experience collecting
                ######
                if reinforce_config.collecting_experience:
                    if previous_state_1 is not None and previous_state_2 is not None and previous_action_1 is not None and previous_action_2 is not None:
                        previous_state_1[5:9] = previous_state_2[0:4] # Include player 2's action
#                         print(previous_state_1[env.miner_index])
                        denorm_previous_state_1 = previous_state_1
                        denorm_previous_state_1[env.miner_index] += denorm_previous_state_1[3] * 50 + 100
#                         print(previous_state_1[env.miner_index])

                        experience = [
                            denorm_previous_state_1,
                            np.append(state_1, previous_reward_1)
                        ]
                        
                        #print(experience)
                        all_experiences.append(experience)
                        if ((len(all_experiences)) % 100 == 0) and reinforce_config.collecting_experience:
                            torch.save(all_experiences, 'abp/examples/pysc2/tug_of_war/all_experience.pt')
#                         pretty_print(len(all_experiences) - 1, all_experiences)
#                         print()
#                         input("pause")
                        
                    previous_state_1 = deepcopy(combine_states_1[choice_1])
                    previous_state_2 = deepcopy(combine_states_2[choice_2])
                env.step(list(actions_1[choice_1]), 1)
#                 input('stepped with command 1')
                env.step(list(actions_2[choice_2]), 2)
                previous_action_1 = deepcopy(actions_1[choice_1])
                previous_action_2 = deepcopy(actions_2[choice_2])
                while skiping:
#                     print("Get actions time:")
#                     start_time = time.time()
                    state_1, state_2, done, dp = env.step([], 0)
                    #input(' step wating for done signal')
                    if dp or done:
#                         print(time.time() - start_time)
                        break
#                 input('done stepping after collecting experience')
                current_reward_1 = 0
                reward_1, reward_2 = env.sperate_reward(env.decomposed_rewards)
                for r1 in reward_1:
                    current_reward_1 += r1
                    
                total_reward_1 += current_reward_1
                previous_reward_1 = current_reward_1

            if reinforce_config.collecting_experience:
                previous_state_1[5:9] = previous_state_2[0:4] # Include player 2's action
                denorm_previous_state_1 = previous_state_1
                denorm_previous_state_1[env.miner_index] += denorm_previous_state_1[3] * 50 + 100
#                         print(previous_state_1[env.miner_index])

                experience = [
                    denorm_previous_state_1,
                    np.append(state_1, previous_reward_1)
                ]
                all_experiences.append(experience)
                if ((len(all_experiences)) % 100 == 0) and reinforce_config.collecting_experience:
                    torch.save(all_experiences, 'abp/examples/pysc2/tug_of_war/all_experience.pt')
#                 pretty_print(len(all_experiences) - 1, all_experiences)
#                 print()
#                 input("pause")

            total_rewwards_list.append(total_reward_1)
            test_summary_writer.add_scalar(tag="Test/Episode Reward", scalar_value=total_reward_1,
                                           global_step=episode + 1)
            test_summary_writer.add_scalar(tag="Test/Steps to choosing Enemies", scalar_value=steps + 1,
                                           global_step=episode + 1)
#         if reinforce_config.collecting_experience:
#             break
        #print(test.size())
        tr = sum(total_rewwards_list) / evaluation_config.test_episodes
        print("total reward:")
        print(tr)
        privous_result.append(tr)
        if len(privous_result) > update_wins_waves:
            del privous_result[0]
            
        f = open("result_self_play.txt", "a+")
        f.write(str(tr) + "\n")
        f.close()
        if not reinforce_config.is_random_agent_1:
            agent_1.enable_learning()
            
#         if not reinforce_config.is_random_agent_2:
#             agent_2.enable_learning()
        
#     if reinforce_config.collecting_experience:
#         torch.save(all_experiences, 'abp/examples/pysc2/tug_of_war/all_experiences_2.pt')
        
def pretty_print(state,  text = ""):
    state_list = state.copy().tolist()
    state = []
    for s in state_list:
        state.append(str(s))
    print("===========================================")
    print(text)
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