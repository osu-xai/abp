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
from sc2env.xai_replay.recorder.recorder import XaiReplayRecorder

def run_task(evaluation_config, network_config, reinforce_config, map_name = None, train_forever = False):
    flags.FLAGS(sys.argv[:1])
    
    reward_types = ['killEnemyMarine',
                    'killEnemyViking',
                    'killEnemyColossus',
                    'newMineralIncome',
                    'numberOfMarinesPerWave',
                    'numberOfVikingsPerWave',
                    'numberOfColossusPerWave',
                    'damageToEnemyBaseHP',
                    'damageToEnemyBaseSheild',
                    'damageToSelfBaseHP_Neg',
                    'damageToSelfBaseSheild_Neg',
                    'win',
                    'loss']
    
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
                        network_config = network_config, # have not done yet
                        reinforce_config = reinforce_config) # have not done yet


    training_summaries_path = evaluation_config.summaries_path + "/train"
    clear_summary_path(training_summaries_path)
    train_summary_writer = SummaryWriter(training_summaries_path)

    test_summaries_path = evaluation_config.summaries_path + "/test"
    clear_summary_path(test_summaries_path)
    test_summary_writer = SummaryWriter(test_summaries_path)

    totalRewardsDict = {}

    skip_steps = 4

    for rt in reward_types:
    	totalRewardsDict['total' + rt] = 0
    
    for episode in range(evaluation_config.training_episodes):
    #for episode in range(1):

        print("=======================================================================")
        print("===============================Now training============================")
        print("=======================================================================")
        print("Now training.")
        state = env.reset()
        total_reward = 0
        end = False
        deciding = True
        steps = 0
        
        while deciding:
            stepRewards = {}
            steps += 1
            action, _, _ = agent.predict(state)
            print("training step " + str(steps))
            #print(action)
            #time.sleep(0.5)
            state, done, dead = env.step(action)
            np.set_printoptions(precision = 2)
            #print(np.reshape(state, (8,40,40)))
            #input("pause")
            for _ in range(skip_steps):
                state, end = env.step(4, skip = True)
                if end:
                    break

            #Skip frames to deal with credit assignment problem
            for i, rt in enumerate(reward_types):
                stepRewards[rt] = env.decomposed_rewards[i]
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
            if end:
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
    '''
    if train_forever:
    	for i in range(10000):
    		agent.update()
    		if i % 100 == 0:
    			print(i)
    '''
    agent.disable_learning(save = False)

    total_rewwards_list = []
    # Test Episodes
    for episode in range(evaluation_config.test_episodes):
    #for episode in range(1):
        print("======================================================================")
        print("===============================Now testing============================")
        print("======================================================================")
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
            
            
            print(action)
            #print(q_values)
            
            if evaluation_config.generate_xai_replay:
                recorder.record_decision_point(action, q_values, combined_q_values, reward, env.decomposed_reward_dict)

            #if evaluation_config.render:
                # env.render()
                # pdx_explanation.render_all_pdx(action, 4, q_values,
                #                                ['Top_Left', 'Top_Right', 'Bottom_Left', 'Bottom_Right'],
                #                                sorted(reward_types))
                
               # time.sleep(evaluation_config.sleep)
                #input("pause")
            
            state, done, dead = env.step(action)
            for _ in range(skip_steps):
                if evaluation_config.generate_xai_replay:
                    recorder.record_game_clock_tick(env.decomposed_reward_dict)
                state, end = env.step(4, skip = True)
                if end:
                    break
            
            if evaluation_config.generate_xai_replay:
                for i in range(5):
                    recorder.record_game_clock_tick(env.decomposed_reward_dict)
                    env.step(action)

            if end:
                if evaluation_config.generate_xai_replay:
                    for i in range(5):
                        recorder.record_game_clock_tick(env.decomposed_reward_dict)
                        env.step(action)
                    recorder.done_recording()
                break

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
