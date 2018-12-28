import gym
import time
import numpy as np

from abp import HRAAdaptive
from abp.utils import clear_summary_path
from abp.explanations import PDX
from tensorboardX import SummaryWriter
from gym.envs.registration import register
from sc2env.environments.four_towers_multi_unit import FourTowersSequentialMultiUnitEnvironment
from sc2env.xai_replay.recorder.recorder import XaiReplayRecorder

from absl import app
from absl import flags
from collections import namedtuple
import datetime
import time
import sys
import numpy as np
import pandas as pd
import csv
import json

def run_task(evaluation_config, network_config, reinforce_config):
    env = FourTowersSequentialMultiUnitEnvironment(evaluation_config.generate_xai_replay)
    
    max_episode_steps = 100
    state = env.reset()
    # print(state)
    choices = [0,1,2,3]
    pdx_explanation = PDX()

    reward_types = ['damageToZealot', 'damageToZergling', 'damageToRoach', 'damageToStalker', 'damageToMarine', 'damageToHydralisk']

    agent = HRAAdaptive(name = "FourTowerSequential",
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

    totalDamageToZealot = 0
    totalDamageToZergling = 0
    totalDamageToRoach = 0
    totalDamageToStalker = 0
    totalDamageToMarine = 0
    totalDamageToHydralisk = 0

    # Training Episodes

    for episode in range(2):
    #for episode in range(evaluation_config.training_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        dead = False
        deciding = True
        running = True
        steps = 0
        rewards = []

        initial_state = np.array(state)

        while deciding:
            steps += 1
            action, q_values, _ = agent.predict(state)
            state, reward, done, dead, info = env.step(action)
            while running:
                action = 4
                state, reward, done, dead, info = env.step(action)
                if done:
                    break

            if not dead:
                rewards = {'damageToZealot': env.decomposed_rewards[len(env.decomposed_rewards) - 1][0], 'damageToZergling': env.decomposed_rewards[len(env.decomposed_rewards) - 1][1], 'damageToRoach': env.decomposed_rewards[len(env.decomposed_rewards) - 1][2], 'damageToStalker': env.decomposed_rewards[len(env.decomposed_rewards) - 1][3], 'damageToMarine': env.decomposed_rewards[len(env.decomposed_rewards) - 1][4], 'damageToHydralisk': env.decomposed_rewards[len(env.decomposed_rewards) - 1][5]}

            else:
                rewards = {'damageToZealot': env.decomposed_rewards[len(env.decomposed_rewards) - 2][0], 'damageToZergling': env.decomposed_rewards[len(env.decomposed_rewards) - 2][1], 'damageToRoach': env.decomposed_rewards[len(env.decomposed_rewards) - 2][2], 'damageToStalker': env.decomposed_rewards[len(env.decomposed_rewards) - 2][3], 'damageToMarine': env.decomposed_rewards[len(env.decomposed_rewards) - 2][4], 'damageToHydralisk': env.decomposed_rewards[len(env.decomposed_rewards) - 2][5]}


            for reward_type in rewards.keys():
                agent.reward(reward_type, rewards[reward_type])

            total_reward += rewards['damageToZealot'] + rewards['damageToZergling'] + rewards['damageToRoach'] + rewards['damageToStalker'] + rewards['damageToMarine'] + rewards['damageToHydralisk']

            if dead:
                break

        totalDamageToZealot += rewards['damageToZealot']
        totalDamageToZergling += rewards['damageToZergling']
        totalDamageToRoach += rewards['damageToRoach']
        totalDamageToStalker += rewards['damageToStalker']
        totalDamageToMarine += rewards['damageToMarine']
        totalDamageToHydralisk += rewards['damageToHydralisk']

        print("Damage to Zealot: {}".format(totalDamageToZealot))
        print("Damage to Zergling: {}".format(totalDamageToZergling))
        print("Damage to Roach: {}".format(totalDamageToRoach))
        print("Damage to Stalker: {}".format(totalDamageToStalker))
        print("Damage to Marine: {}".format(totalDamageToMarine))
        print("Damage to Hydralisk: {}".format(totalDamageToHydralisk))

        agent.end_episode(state)
        test_summary_writer.add_scalar(tag="Train/Episode Reward", scalar_value=total_reward,
                                       global_step=episode + 1)
        train_summary_writer.add_scalar(tag="Train/Steps to collect all Fruits", scalar_value=steps + 1,
                                        global_step=episode + 1)

        print("EPISODE REWARD {}".format(total_reward))
        print("EPISODE {}".format(episode))

    agent.disable_learning()

    # Test Episodes
    # for now, pull the instantiation of this out of the loop, so that we get video with multiple episodes
    if evaluation_config.generate_xai_replay:
        recorder = XaiReplayRecorder(env.sc2_env)
    #for episode in range(evaluation_config.test_episodes):
    for episode in range(1):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        deciding = True
        running = True
        reward = [[]]
        #recorder = XaiReplayRecorder(env.sc2_env)
        while deciding:
            steps += 1
            action, q_values, combined_q_values = agent.predict(state)
            # print("Jed here")
            # print(action)
            # print(q_values)
            if evaluation_config.generate_xai_replay:
                recorder.record_decision_point(state, action, q_values, combined_q_values, reward)
            # if evaluation_config.render:
            #     # env.render()
            #     pdx_explanation.render_all_pdx(action, 4, q_values, ['Top_Left', 'Top_Right', 'Bottom_Left', 'Bottom_Right'], ['damageToZealot', 'damageToZergling', 'damageToRoach', 'damageToStalker', 'damageToMarine', 'damageToHydralisk'])
            #     time.sleep(evaluation_config.sleep)
            #     # This renders an image of the game and saves to test.jpg

            state, reward, done, dead, info = env.step(action)
            print("state :{}".format(state))
            print(f"done  :{done}")
            print("reward :{}".format(reward))
            print("dead :{}".format(dead))
            print("info :{}".format(info))
            while running:
                action = 4
                if evaluation_config.generate_xai_replay:
                    recorder.record_game_clock_tick(state, reward)
                state, reward, done, dead, info = env.step(action)
                if done:
                    break

            if dead:
                if evaluation_config.generate_xai_replay:
                    recorder.done_recording()
                break

        agent.end_episode(state)

        test_summary_writer.add_scalar(tag="Test/Episode Reward", scalar_value=total_reward,
                                       global_step=episode + 1)
        test_summary_writer.add_scalar(tag="Test/Steps to collect all Fruits", scalar_value=steps + 1,
                                       global_step=episode + 1)
