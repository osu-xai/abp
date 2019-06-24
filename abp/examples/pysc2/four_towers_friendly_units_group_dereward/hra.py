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
from sc2env.environments.four_towers_friendly_units_group_dereward import FourTowersFriendlyUnitsGroupDereward
from sc2env.xai_replay.recorder.recorder import XaiReplayRecorder
from tqdm import tqdm

def run_task(evaluation_config, network_config, reinforce_config, map_name = None, train_forever = False):
    flags.FLAGS(sys.argv[:1])
    
    reward_types = ['damageToWeakEnemyGroup',
                    'destoryToWeakEnemyGroup',
                    'damageToStrongEnemyGroup',
                    'destoryToStrongEnemyGroup',
                    'damageToWeakFriendGroup',
                    'destoryToWeakFriendGroup',
                    'damageToStrongFriendGroup',
                    'destoryToStrongFriendGroup']
    
    ec = evaluation_config
    replay_dimension = ec.xai_replay_dimension
    env = FourTowersFriendlyUnitsGroupDereward(reward_types, map_name = map_name, unit_type = [83, 48, 52], \
        generate_xai_replay = ec.generate_xai_replay, xai_replay_dimension = replay_dimension)
    
    max_episode_steps = 500
    state = env.reset()
    
    choices = [0,1,2,3]
    pdx_explanation = PDX()

    agent = HRAAdaptive(name = "FourTowersFriendlyUnitsGroupDereward",
                        choices = choices,
                        reward_types = reward_types,
                        network_config = network_config,
                        reinforce_config = reinforce_config)


    training_summaries_path = evaluation_config.summaries_path + "/train"
    clear_summary_path(training_summaries_path)
    train_summary_writer = SummaryWriter(training_summaries_path)

    test_summaries_path = evaluation_config.summaries_path + "/test"
    clear_summary_path(test_summaries_path)
    test_summary_writer = SummaryWriter(test_summaries_path)

    totalRewardsDict = {}

    for rt in reward_types:
    	totalRewardsDict['total' + rt] = 0
    print("=======================================================================")
    print("===============================Now training============================")
    print("=======================================================================")
    print("Now training.")
    for episode in tqdm(range(evaluation_config.training_episodes)):
    #for episode in range(1):

        state = env.reset()
        total_reward = 0
        done = False
        dead = False
        deciding = True
        running = True
        steps = 0
        
        while deciding and steps < max_episode_steps:
            stepRewards = {}
            steps += 1
            action, q_values,combined_q_values = agent.predict(state)
            state, done, dead = env.step(action)
            while running:
                action = 4
                state, done, dead = env.step(action)
                if done:
                    break

            state, _, _ = env.step(4)
            for i, rt in enumerate(reward_types):
                stepRewards[rt] = env.decomposed_rewards[len(env.decomposed_rewards) - 1][i]
                agent.reward(rt, stepRewards[rt])
                total_reward += stepRewards[rt]
            if dead:
                break
        for i in range(len(totalRewardsDict)):
            totalRewardsDict[list(totalRewardsDict.keys())[i]] += stepRewards[reward_types[i]]

        agent.end_episode(env.end_state)
        test_summary_writer.add_scalar(tag = "Train/Episode Reward", scalar_value = total_reward,
                                       global_step = episode + 1)
        train_summary_writer.add_scalar(tag = "Train/Steps to choosing Enemies", scalar_value = steps + 1,
                                        global_step = episode + 1)

    '''
    if train_forever:
    	for i in range(10000):
    		agent.update()
    		if i % 100 == 0:
    			print(i)
    '''
    agent.disable_learning(save = False)
    print("======================================================================")
    print("===============================Now testing============================")
    print("======================================================================")
    total_rewwards_list = []
    # Test Episodes
    for episode in tqdm(range(evaluation_config.test_episodes)):
#     for episode in range(1):

        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        deciding = True
        running = True
        reward = [[]]
        
        if evaluation_config.generate_xai_replay:
            recorder = XaiReplayRecorder(env.sc2_env, episode, evaluation_config.env, ['Top_Left', 'Top_Right', 'Bottom_Left', 'Bottom_Right'], sorted(reward_types), replay_dimension)

        while deciding:
            #input("pause")
            steps += 1
            action, q_values,combined_q_values = agent.predict(state)
            
            print(action)
            input("pause")
#             print(action)
            #print(q_values)
            
            if evaluation_config.generate_xai_replay:
                recorder.record_decision_point(action, q_values, combined_q_values, reward, env.decomposed_reward_dict)

            if evaluation_config.render:
                # env.render()
                pdx_explanation.render_all_pdx(action, 4, q_values,
                                               ['Top_Left', 'Top_Right', 'Bottom_Left', 'Bottom_Right'],
                                               sorted(reward_types))
                
               # time.sleep(evaluation_config.sleep)
            
            
            state, done, dead = env.step(action)

            while running:
                action = 4
                if evaluation_config.generate_xai_replay:
                    recorder.record_game_clock_tick(env.decomposed_reward_dict)
                    #print(env.decomposed_reward_dict)
                state, done, dead = env.step(action)
                if done:
                    break
            
            if evaluation_config.generate_xai_replay:
                for i in range(5):
                    recorder.record_game_clock_tick(env.decomposed_reward_dict)
                    env.step(action)

            if dead:
                if evaluation_config.generate_xai_replay:
                    for i in range(5):
                        recorder.record_game_clock_tick(env.decomposed_reward_dict)
                        env.step(action)
                    recorder.done_recording()
                break
            state, _, _ = env.step(4)
            
            for i, rt in enumerate(reward_types):
                total_reward += env.decomposed_rewards[len(env.decomposed_rewards) - 1][i]
        
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
