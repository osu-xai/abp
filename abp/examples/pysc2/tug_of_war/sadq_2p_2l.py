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

# np.set_printoptions(precision = 2)
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
        
#     if not reinforce_config.is_random_agent_2:
#         agent_2 = SADQAdaptive(name = "TugOfWar",
#                             state_length = len(state_2),
#                             network_config = network_config,
#                             reinforce_config = reinforce_config)
#         print("sadq agent 2")
#     else:
#         print("random agent 2")
        
    training_summaries_path = evaluation_config.summaries_path + "/train"
    clear_summary_path(training_summaries_path)
    train_summary_writer = SummaryWriter(training_summaries_path)

    test_summaries_path = evaluation_config.summaries_path + "/test"
    clear_summary_path(test_summaries_path)
    test_summary_writer = SummaryWriter(test_summaries_path)
    
    agents_2 = ["random"]
    ###############test###################
#     new_agent_2 = SADQAdaptive(name = "TugOfWar_test1",
#             state_length = len(state_2),
#             network_config = network_config,
#             reinforce_config = reinforce_config)

#     new_agent_2.load_model(agent_1.eval_model)
#     new_agent_2.disable_learning(is_save = False)
#     agents_2.append(new_agent_2)
#     new_agent_2 = SADQAdaptive(name = "TugOfWar_test2",
#             state_length = len(state_2),
#             network_config = network_config,
#             reinforce_config = reinforce_config)

#     new_agent_2.load_model(agent_1.eval_model)
#     new_agent_2.disable_learning(is_save = False)
#     agents_2.append(new_agent_2)
    ######################################
#     random_enemy = True
    
    round_num = 0
    
    privous_result = []
    update_wins_waves = 10
    
    all_experiences = []
    
    
    
#     if not reinforce_config.is_random_agent_2:
#         agent_2.load_model(agent_1.eval_model)
        
    while True:
        if len(privous_result) >= update_wins_waves and \
        sum(privous_result) / update_wins_waves > 11000 and \
        not reinforce_config.is_random_agent_2:
            privous_result = []
            print("replace enemy agent's weight with self agent")
#             random_enemy = False
            f = open("result_self_play_2l.txt", "a+")
            f.write("Update agent\n")
            f.close()
            
            new_agent_2 = SADQAdaptive(name = "TugOfWar_" + str(round_num),
                    state_length = len(state_2),
                    network_config = network_config,
                    reinforce_config = reinforce_config)
            
            new_agent_2.load_model(agent_1.eval_model)
            new_agent_2.disable_learning(is_save = False)
            agents_2.append(new_agent_2)
            
            agent_1.steps = reinforce_config.epsilon_timesteps / 2
            agent_1.best_reward_mean = 0
            agent_1.save(force = True, appendix = "update_" + str(round_num))
            
        round_num += 1
    
        print("=======================================================================")
        print("===============================Now training============================")
        print("=======================================================================")
        print("Now training.")
        
        print("Now have {} enemy".format(len(agents_2)))
        
        for idx_enemy, enemy_agent in enumerate(agents_2[::-1]):
#             break
            if reinforce_config.collecting_experience:
                break
            if enemy_agent == "random":
                print(enemy_agent)
            else:
                print(enemy_agent.name)
                
            if idx_enemy == 0:
                training_num = evaluation_config.training_episodes
            else:
                training_num = 10
                
            for episode in tqdm(range(training_num)):
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
    #                 print("=======================================================================")
    #                 pretty_print(state_1, text = "state 1")
    #                 pretty_print(state_2, text = "state 2")

                    actions_1 = env.get_big_A(state_1[env.miner_index], state_1[env.pylon_index], is_train = True)
                    actions_2 = env.get_big_A(state_2[env.miner_index], state_2[env.pylon_index], is_train = True)
                    assert state_1[-1] == state_2[-1] == steps, print(state_1, state_2, steps)
                    if not reinforce_config.is_random_agent_1:
                        combine_states_1 = combine_sa(state_1, actions_1)
    #                     print(combine_states_1)
    #                     print(env.normalization(combine_states_1))
    #                     print(state_1[env.miner_index])
                        choice_1, _ = agent_1.predict(env.normalization(combine_states_1))
    #                     input()
    #                     for cs1 in combine_states_1:
    #                         print(cs1.tolist())
                    else:
    #                     combine_states_1 = combine_sa(state_1, actions_1)
                        choice_1 = randint(0, len(actions_1) - 1)

                    if not reinforce_config.is_random_agent_2 and enemy_agent != "random":
                        combine_states_2 = combine_sa(state_2, actions_2)
                        choice_2, _ = enemy_agent.predict(env.normalization(combine_states_2))
                    else:
                        choice_2 = randint(0, len(actions_2) - 1)
    #                 print("action list:")
    #                 print(actions_1)
    #                 print(actions_2)
    #                 assign action
    #                 print("choice:")
    #                 print(actions_1[choice_1])
    #                 print(actions_2[choice_2])
    #                 pretty_print(combine_states_1[choice_1], text = "after state:")
    #                 input("pause")
    #                 print(combine_states_2[choice_2].tolist())
    #                 if state_1[env.miner_index] > 300:
    #                     input('pause')
                    env.step(list(actions_1[choice_1]), 1)
                    env.step(list(actions_2[choice_2]), 2)
    #                 if steps == 1:
    #                     env.step((1,0,0,0,0,0,0), 1)
    #                     env.step((1,0,0,0,0,0,0), 2)
    #                 if steps > 2:
    #                     env.step((1,0,0,0,0,0,0), 1)
    #                     env.step((0,0,0,0,0,0,1), 2)
                    while skiping:
                        state_1, state_2, done, dp = env.step([], 0)
    #                     input('time_step')
                        if dp or done:
                            break
                    if steps == max_episode_steps or done:
                        win_lose = player_1_win_condition(state_1[27], state_1[28], state_1[29], state_1[30])

                        if win_lose == 1:
                            env.decomposed_rewards[4] = 10000
                        elif win_lose == -1:
                            env.decomposed_rewards[5] = 10000

                    reward_1, reward_2 = env.sperate_reward(env.decomposed_rewards)
    #                 print('reward:')
    #                 print(state_1[27], state_1[28], state_1[29], state_1[30])
    #                 print(reward_1)
    #                 print(reward_2)
    #                 if steps == max_episode_steps or done:
    #                     input()

                    if not reinforce_config.is_random_agent_1:
                        agent_1.reward(sum(reward_1))

                if not reinforce_config.is_random_agent_1:
                    agent_1.end_episode(state_1)

                test_summary_writer.add_scalar(tag = "Train/Episode Reward", scalar_value = total_reward,
                                               global_step = episode + 1)
                train_summary_writer.add_scalar(tag = "Train/Steps to choosing Enemies", scalar_value = steps + 1,
                                                global_step = episode + 1)
        
        if not reinforce_config.is_random_agent_1:
            agent_1.disable_learning(is_save = not reinforce_config.collecting_experience)

        total_rewwards_list = []
            
        # Test Episodes
        print("======================================================================")
        print("===============================Now testing============================")
        print("======================================================================")
                
        tied_lose = 0
        for idx_enemy, enemy_agent in enumerate(agents_2[::-1]):
            average_end_state = np.zeros(len(state_1))
            if enemy_agent == "random":
                print(enemy_agent)
            else:
                print(enemy_agent.name)
                
            if idx_enemy == 0:
                test_num = evaluation_config.test_episodes
            else:
                test_num = 5
                
            for episode in tqdm(range(test_num)):
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



    #                 print(time.time() - start_time)


                    if not reinforce_config.is_random_agent_1:
                        actions_1 = env.get_big_A(state_1[env.miner_index], state_1[env.pylon_index])
                        combine_states_1 = combine_sa(state_1, actions_1)
                        choice_1, _ = agent_1.predict(env.normalization(combine_states_1))
                    else:
                        actions_1 = env.get_big_A(state_1[env.miner_index], state_1[env.pylon_index], is_train = True)
                        combine_states_1 = combine_sa(state_1, actions_1)
                        choice_1 = randint(0, len(actions_1) - 1)


                    if not reinforce_config.is_random_agent_2 and enemy_agent != "random":
                        actions_2 = env.get_big_A(state_2[env.miner_index], state_2[env.pylon_index])
                        combine_states_2 = combine_sa(state_2, actions_2)
                        choice_2, _ = enemy_agent.predict(env.normalization(combine_states_2))
                    else:
                        actions_2 = env.get_big_A(state_2[env.miner_index], state_2[env.pylon_index], is_train = True)
                        combine_states_2 = combine_sa(state_2, actions_2)
                        choice_2 = randint(0, len(actions_2) - 1)

    #                 input('stepped with command 2')
                    #######
                    #experience collecting
                    ######
                    if reinforce_config.collecting_experience:
                        if previous_state_1 is not None and previous_state_2 is not None and previous_action_1 is not None and previous_action_2 is not None:
                            previous_state_1[1:7] = previous_state_2[8:14] # Include player 2's action
                            previous_state_1[env.miner_index] += previous_state_1[env.pylon_index] * 75 + 100

                            experience = [
                                previous_state_1,
                                np.append(state_1, previous_reward_1)
                            ]
                            all_experiences.append(experience)
                            if ((len(all_experiences)) % 100 == 0) and reinforce_config.collecting_experience:
                                torch.save(all_experiences, 'abp/examples/pysc2/tug_of_war/test_random_vs_random_2l.pt')

                        previous_state_1 = deepcopy(combine_states_1[choice_1])
                        previous_state_2 = deepcopy(combine_states_2[choice_2])

                        previous_action_1 = deepcopy(actions_1[choice_1])
                        previous_action_2 = deepcopy(actions_2[choice_2])

                    env.step(list(actions_1[choice_1]), 1)
                    env.step(list(actions_2[choice_2]), 2)

                    while skiping:
    #                     print("Get actions time:")
    #                     start_time = time.time()
                        state_1, state_2, done, dp = env.step([], 0)
                        #input(' step wating for done signal')
                        if dp or done:
    #                         print(time.time() - start_time)
                            break
    #                 input('done stepping after collecting experience')
    #                 current_reward_1 = 0

                    if steps == max_episode_steps or done:
                        win_lose = player_1_win_condition(state_1[27], state_1[28], state_1[29], state_1[30])

                        if win_lose == 1:
                            env.decomposed_rewards[4] = 10000
                        elif win_lose == -1:
                            env.decomposed_rewards[5] = 10000

                    reward_1, reward_2 = env.sperate_reward(env.decomposed_rewards)
    #                 print(env.decomposed_rewards)
    #                 print(reward_1, reward_2)

    #                 for r1 in reward_1:
                    current_reward_1 = sum(reward_1)
    #                 print(current_reward_1)


                    total_reward_1 += current_reward_1
    #                 print(total_reward_1)
    #                 if total_reward_1 > 14000 or total_reward_1 < -14000:
    #                     input()
                    previous_reward_1 = current_reward_1

                if reinforce_config.collecting_experience:
                    previous_state_1[1:7] = previous_state_2[8:14] # Include player 2's action
                    previous_state_1[env.miner_index] += previous_state_1[env.pylon_index] * 75 + 100

                    experience = [
                        previous_state_1,
                        np.append(state_1, previous_reward_1)
                    ]
                    all_experiences.append(experience)
                    if ((len(all_experiences)) % 100 == 0) and reinforce_config.collecting_experience:
                        torch.save(all_experiences, 'abp/examples/pysc2/tug_of_war/test_random_vs_random_2l.pt')

                average_end_state += state_1

                total_rewwards_list.append(total_reward_1)
                test_summary_writer.add_scalar(tag="Test/Episode Reward", scalar_value=total_reward_1,
                                               global_step=episode + 1)
                test_summary_writer.add_scalar(tag="Test/Steps to choosing Enemies", scalar_value=steps + 1,
                                               global_step=episode + 1)
    #         if reinforce_config.collecting_experience:
    #             break
            #print(test.size())
    #         print(total_rewwards_list)
            total_rewards_list_np = np.array(total_rewwards_list)

            tied = np.sum(total_rewards_list_np[-test_num:] == 0)
            wins = np.sum(total_rewards_list_np[-test_num:] > 0)
            lose = np.sum(total_rewards_list_np[-test_num:] < 0)
            
            tied_lose += (tied + lose)
            print("wins/lose/tied")
            print(str(wins / test_num * 100) + "% \t",
                 str(lose / test_num * 100) + "% \t",
                 str(tied / test_num * 100) + "% \t")
            pretty_print(average_end_state / test_num)
            
            
        tr = sum(total_rewwards_list) / len(total_rewwards_list)
        print("total reward:")
        print(tr)
        
        privous_result.append(tr)
        
        if len(privous_result) > update_wins_waves:
            del privous_result[0]
        f = open("result_self_play_2l.txt", "a+")
        f.write(str(tr) + "\n")
        f.close()
        
        if tied_lose == 0:
            agent_1.save(force = True, appendix = "_the_best")
            
        if not reinforce_config.is_random_agent_1:
            agent_1.enable_learning()
            
def player_1_win_condition(state_1_T_hp, state_1_B_hp, state_2_T_hp, state_2_B_hp):
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