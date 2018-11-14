import gym
import time
import numpy as np
from absl import flags
import sys

from abp import HRAAdaptive
from abp.utils import clear_summary_path
from abp.explanations import PDX
from tensorboardX import SummaryWriter
from gym.envs.registration import register
from abp.openai.envs.four_towers_friendly_units.FourTowerSequentialFriendlyUnits import FourTowerSequentialFriendlyUnits

def run_task(evaluation_config, network_config, reinforce_config):
    flags.FLAGS(sys.argv[:1])
    env = FourTowerSequentialFriendlyUnits()
    
    # env = gym.make(evaluation_config.env)
    max_episode_steps = 100
    state = env.reset()
 #   time.sleep(100000)
    # print(state)
    
    choices = [0,1,2,3]
    pdx_explanation = PDX()

    reward_types = ['damageToMarine',
                    'damageByMarine',
                    'damageToZergling',
                    'damageByZergling',
                    'damageToMarauder',
                    'damageByMarauder',
                    'damageToHydralisk',
                    'damageByHydralisk',
                    'damageToThor',
                    'damageByThor',
                    'damageToUltralisk',
                    'damageByUltralisk',
                    'penalty']

    agent = HRAAdaptive(name = "FourTowerSequentialFriendlyUnits",
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


    totalRewardsDict = {
        'totalDamageToMarine' : 0,
        'totalDamageByMarine' : 0,
        'totalDamageToZergling' : 0,
        'totalDamageByZergling' : 0,
        'totalDamageToMarauder' : 0,
        'totalDamageByMarauder' : 0,
        'totalDamageToHydralisk' : 0,
        'totalDamageByHydralisk' : 0,
        'totalDamageToThor' : 0,
        'totalDamageByThor' : 0,
        'totalDmageToUltralisk' : 0,
        'totalDamageByUltralisk' : 0,
        'totalPenalty' : 0
        }
    
    # Training Episodes
    
    for episode in range(evaluation_config.training_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        dead = False
        deciding = True
        running = True
        steps = 0
        
 #       initial_state = np.array(state)
 #       print(np.array(state))
        

 #       time.sleep(0.2)
        while deciding:
            rewards = {}
            steps += 1
            action, q_values = agent.predict(np.array(state))
 #           print(action)
            #time.sleep(0.5)
            state, done, dead = env.step(action)
            while running:
                action = 4
                state, done, dead = env.step(action)
                if done:
                    break
            
            for i, rt in enumerate(reward_types):
                rewards[rt] = env.decomposed_rewards[len(env.decomposed_rewards) - 1][i]
                agent.reward(rt, rewards[rt])
                total_reward += rewards[rt]
 #           print("l1:")
#            print(rewards)
 #           print(np.array(env.decomposed_rewards))
#            print(rewards)
 #           time.sleep(15)
            if dead:
                break
 #       print(np.array(env.decomposed_rewards))
#        time.sleep(10)
        for i in range(len(totalRewardsDict)):
            totalRewardsDict[list(totalRewardsDict.keys())[i]] += rewards[reward_types[i]]

        agent.end_episode(np.array(state))
        test_summary_writer.add_scalar(tag = "Train/Episode Reward", scalar_value = total_reward,
                                       global_step = episode + 1)
        train_summary_writer.add_scalar(tag = "Train/Steps to choosing Enemies", scalar_value = steps + 1,
                                        global_step = episode + 1)

        print("EPISODE REWARD {}".format(total_reward))
        print("EPISODE {}".format(episode))
        
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
        
        while deciding:
            steps += 1
            action, q_values = agent.predict(np.array(state))
            print(action)
            print(q_values)
            
            if evaluation_config.render:
                # env.render()
                pdx_explanation.render_all_pdx(action, 4, q_values,
                                               ['Top_Left', 'Top_Right', 'Bottom_Left', 'Bottom_Right'],
                                               ['damageToMarine',
                                                'damageByMarine',
                                                'damageToZergling',
                                                'damageByZergling',
                                                'damageToMarauder',
                                                'damageByMarauder',
                                                'damageToHydralisk',
                                                'damageByHydralisk',
                                                'damageToThor',
                                                'damageByThor',
                                                'damageToUltralisk',
                                                'damageByUltralisk',
                                                'penalty'])
                
                time.sleep(evaluation_config.sleep)
            
                
            state, done, dead = env.step(action)

            while running:
#                action = 4
                state, done, dead = env.step(action)
                if done:
                    break

            if dead:
                break
            for i, rt in enumerate(reward_types):
                total_reward += env.decomposed_rewards[len(env.decomposed_rewards) - 1][i]
        
        total_rewwards_list.append(total_reward)
        
        agent.end_episode(np.array(state))

        test_summary_writer.add_scalar(tag="Test/Episode Reward", scalar_value=total_reward,
                                       global_step=episode + 1)
        test_summary_writer.add_scalar(tag="Test/Steps to choosing Enemies", scalar_value=steps + 1,
                                       global_step=episode + 1)
        
    print(sum(total_rewwards_list) / evaluation_config.test_episodes)
