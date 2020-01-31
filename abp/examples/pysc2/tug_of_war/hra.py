import gym
import time
import numpy as np
from absl import flags
import sys, os

from abp import HRAAdaptive
from abp.utils import clear_summary_path
from abp.explanations import PDX
from tensorboardX import SummaryWriter
from gym.envs.registration import register
from sc2env.environments.tug_of_war import TugOfWar
#from sc2env.xai_replay.recorder.recorder import XaiReplayRecorder
from tqdm import tqdm

np.set_printoptions(precision = 2)
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
    
    ec = evaluation_config
    replay_dimension = ec.xai_replay_dimension
    env = TugOfWar(reward_types, map_name = map_name, \
        generate_xai_replay = ec.generate_xai_replay, xai_replay_dimension = replay_dimension)
    
 #   max_episode_steps = 500
    state = env.reset()
    
    choices = [0,1,2,3,4]
    pdx_explanation = PDX()

    agent = HRAAdaptive(name = "TugOfWar",
                        choices = choices,
                        reward_types = reward_types,
                        network_config = network_config,
                        reinforce_config = reinforce_config,
                        illegal_actions_func = env.get_illegal_actions)


    training_summaries_path = evaluation_config.summaries_path + "/train"
    clear_summary_path(training_summaries_path)
    train_summary_writer = SummaryWriter(training_summaries_path)

    test_summaries_path = evaluation_config.summaries_path + "/test"
    clear_summary_path(test_summaries_path)
    test_summary_writer = SummaryWriter(test_summaries_path)

    totalRewardsDict = {}

    skip_steps = 20

    for rt in reward_types:
    	totalRewardsDict['total' + rt] = 0

    print("=======================================================================")
    print("===============================Now training============================")
    print("=======================================================================")
    print("Now training.")
    for episode in tqdm(range(evaluation_config.training_episodes)):
    #for episode in range(1):
    #    break
        state = env.reset()
        total_reward = 0
        end = False
        deciding = True
        steps = 0
        
        while deciding:
            stepRewards = {}
            steps += 1
            
            #skip frames
            for _ in range(skip_steps):
                state, end, get_income = env.step(4, skip = True)
            # Decision point
            while True:
                action, _, _ = agent.predict(state)
                state, end, get_income = env.step(action)
                if action == 4 or end:
                    break
                for i, rt in enumerate(reward_types):
                    stepRewards[rt] = env.decomposed_rewards[i]
                    agent.reward(rt, stepRewards[rt])
                    total_reward += stepRewards[rt]
                
                print(action)
                print(state)
                print(np.array(env.decomposed_rewards))
                input('pause')
                
            
            #Skip frames to get to next decision point
            while not get_income and not end:
                state, end, get_income = env.step(4, skip = True)
                
            state, end, get_income = env.step(4)
            for i, rt in enumerate(reward_types):
                stepRewards[rt] = env.decomposed_rewards[i]
                agent.reward(rt, stepRewards[rt])
                total_reward += stepRewards[rt]
            print(action)
            print(state)
            print(np.array(env.decomposed_rewards))
            input('pause')
            if end:
                break
        agent.end_episode(env.end_state)        
        for i in range(len(totalRewardsDict)):
            totalRewardsDict[list(totalRewardsDict.keys())[i]] += stepRewards[reward_types[i]]

        test_summary_writer.add_scalar(tag = "Train/Episode Reward", scalar_value = total_reward,
                                       global_step = episode + 1)
        train_summary_writer.add_scalar(tag = "Train/Steps to choosing Enemies", scalar_value = steps + 1,
                                        global_step = episode + 1)

    agent.disable_learning(save = True)
    
    total_rewwards_list = []
    
    # Test Episodes
    print("======================================================================")
    print("===============================Now testing============================")
    print("======================================================================")
    for episode in tqdm(range(evaluation_config.test_episodes)):

        state = env.reset()
        total_reward = 0
        end = False
        deciding = True
        steps = 0
        
        
        # if evaluation_config.generate_xai_replay:
        #     recorder = XaiReplayRecorder(env.sc2_env, episode, evaluation_config.env, ['Top_Left', 'Top_Right', 'Bottom_Left', 'Bottom_Right'], sorted(reward_types), replay_dimension)

        while deciding:
            #input("pause")
            steps += 1
            action, q_values,combined_q_values = agent.predict(state)
            for _ in range(skip_steps):
                state, end, get_income = env.step(4, skip = True)
            while True:
                action, q_values,combined_q_values = agent.predict(state)
                if evaluation_config.generate_xai_replay:
                    recorder.record_decision_point(action, q_values, combined_q_values, reward, env.decomposed_reward_dict)
                state, end, get_income = env.step(action)
                if action == 4 or end:
                    break
                for i, rt in enumerate(reward_types):
                    total_reward += env.decomposed_rewards[i]
#                 print(action)
#                 print(state)
#                 print(np.array(env.decomposed_rewards))
#                 input('pause')
            #if evaluation_config.render:
                # env.render()
                # pdx_explanation.render_all_pdx(action, 4, q_values,
                #                                ['Top_Left', 'Top_Right', 'Bottom_Left', 'Bottom_Right'],
                #                                sorted(reward_types))
                
               # time.sleep(evaluation_config.sleep)
                #input("pause")
            #Skip frames to get to next decision point
            while not get_income and not end:
                if evaluation_config.generate_xai_replay:
                    recorder.record_game_clock_tick(env.decomposed_reward_dict)
                state, end, get_income = env.step(4, skip = True)

            if evaluation_config.generate_xai_replay:
                recorder.record_game_clock_tick(env.decomposed_reward_dict)
            state, end, get_income = env.step(4)

            for i, rt in enumerate(reward_types):
                total_reward += env.decomposed_rewards[i]

#             if evaluation_config.generate_xai_replay:
#                 for i in range(5):
#                     recorder.record_game_clock_tick(env.decomposed_reward_dict)
#                     env.step(action)
#             print(action)
#             print(state)
#             print(np.array(env.decomposed_rewards))
#             input('pause')

            if end:
                if evaluation_config.generate_xai_replay:
                    for i in range(5):
                        recorder.record_game_clock_tick(env.decomposed_reward_dict)
                        env.step(action)
                    recorder.done_recording()
                break
                
        total_rewwards_list.append(total_reward)
        test_summary_writer.add_scalar(tag="Test/Episode Reward", scalar_value=total_reward,
                                       global_step=episode + 1)
        test_summary_writer.add_scalar(tag="Test/Steps to choosing Enemies", scalar_value=steps + 1,
                                       global_step=episode + 1)
    tr = sum(total_rewwards_list) / evaluation_config.test_episodes
    print("total reward:")
    print(tr)
    f = open("result.txt", "a+")
    f.write(str(tr) + "\n")
    f.close()
