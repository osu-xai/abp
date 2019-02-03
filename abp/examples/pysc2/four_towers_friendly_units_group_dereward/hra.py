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

    env = FourTowersFriendlyUnitsGroupDereward(reward_types, map_name = map_name, unit_type = [83, 48, 52])
    
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
    
    for episode in range(evaluation_config.training_episodes):
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
            #print(action)
            #time.sleep(0.5)
            state, done, dead = env.step(action)
            np.set_printoptions(precision = 2)
            #print(np.reshape(state, (8,40,40)))
            #input("pause")
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
            
            #print("l1:")
#            print(rewards)
 #           print(np.array(env.decomposed_rewards))
            #print(rewards)


            #print(stepRewards)
            #print(sum(rewards.values()))
            #time.sleep(40)
            #input("pause")
            if dead:
                break
        for i in range(len(totalRewardsDict)):
            totalRewardsDict[list(totalRewardsDict.keys())[i]] += stepRewards[reward_types[i]]

        agent.end_episode(env.end_state)
        #np.set_printoptions(precision = 2)
        #print(np.reshape(env.end_state, (13,40,40)))
        #print(rewards)
        #time.sleep(40)
        #input("pause")
        test_summary_writer.add_scalar(tag = "Train/Episode Reward", scalar_value = total_reward,
                                       global_step = episode + 1)
        train_summary_writer.add_scalar(tag = "Train/Steps to choosing Enemies", scalar_value = steps + 1,
                                        global_step = episode + 1)

 #       print("EPISODE REWARD {}".format(total_reward))
#        print("EPISODE {}".format(episode))
    if train_forever:
    	for i in range(10000):
    		agent.update()
    		if i % 100 == 0:
    			print(i)
    agent.disable_learning()

    total_rewwards_list = []
    # Test Episodes
    for episode in range(evaluation_config.test_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        deciding = True
        running = True
        ''
        while deciding:
            #input("pause")
            steps += 1
            action, q_values,combined_q_values = agent.predict(state)
            
            print(action)
            print(q_values)
            
            if evaluation_config.render:
                # env.render()
                pdx_explanation.render_all_pdx(action, 4, q_values,
                                               ['Top_Left', 'Top_Right', 'Bottom_Left', 'Bottom_Right'],
                                               sorted(reward_types))
                
               # time.sleep(evaluation_config.sleep)
                input("pause")
            
            state, done, dead = env.step(action)

            while running:
                action = 4
                state, done, dead = env.step(action)
                if done:
                    break

            if dead:
                break
            state, _, _ = env.step(4)
            for i, rt in enumerate(reward_types):
                total_reward += env.decomposed_rewards[len(env.decomposed_rewards) - 1][i]
        
        total_rewwards_list.append(total_reward)
        
        #agent.end_episode(env.last_state)

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
