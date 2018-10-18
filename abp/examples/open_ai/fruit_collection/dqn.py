import time

import gym
from tensorboardX import SummaryWriter

from abp import DQNAdaptive
from abp.utils import clear_summary_path

# TODO: Remove action-on-import, replace with an explicit init
from abp.openai import envs


def run_task(evaluation_config, network_config, reinforce_config):
    start_time = time.time()
    print('Initializing FruitCollection environment...')
    env = gym.make(evaluation_config.env)
    state = env.reset(state_representation="linear")
    LEFT, RIGHT, UP, DOWN = [0, 1, 2, 3]
    choices = [LEFT, RIGHT, UP, DOWN]
    print('Initialized FruitCollection environment in {:.03f} sec'.format(time.time() - start_time))

    start_time = time.time()
    print('Initializing DQNAdaptive agent...')
    agent = DQNAdaptive(name="FruitCollecter",
                        choices=choices,
                        network_config=network_config,
                        reinforce_config=reinforce_config)

    training_summaries_path = evaluation_config.summaries_path + "/train"
    clear_summary_path(training_summaries_path)
    train_summary_writer = SummaryWriter(training_summaries_path)

    test_summaries_path = evaluation_config.summaries_path + "/test"
    clear_summary_path(test_summaries_path)
    test_summary_writer = SummaryWriter(test_summaries_path)
    print('Initialized agent in {:.03f} sec'.format(time.time() - start_time))

    start_time = time.time()
    print('Training for {} episodes...'.format(evaluation_config.training_episodes))
    # Training Episodes
    for episode in range(evaluation_config.training_episodes):
        state = env.reset(state_representation="linear")
        total_reward = 0
        done = False
        steps = 0
        while not done:
            steps += 1
            action, q_values = agent.predict(state)
            state, reward, done, info = env.step(action)

            agent.reward(reward)

            total_reward += reward

        agent.end_episode(state)
        test_summary_writer.add_scalar(tag="Train/Episode Reward",
                                       scalar_value=total_reward,
                                       global_step=episode + 1)

        train_summary_writer.add_scalar(tag="Train/Steps to collect all Fruits",
                                        scalar_value=steps + 1,
                                        global_step=episode + 1)

    print('Completed training in {:.03f} sec'.format(time.time() - start_time))
    agent.disable_learning()

    # Test Episodes
    start_time = time.time()
    print('Testing for {} episodes...'.format(evaluation_config.training_episodes))
    for episode in range(evaluation_config.test_episodes):
        state = env.reset(state_representation="linear")
        total_reward = 0
        done = False
        steps = 0

        while not done:
            steps += 1
            action, q_values = agent.predict(state)
            if evaluation_config.render:
                env.render()
                time.sleep(0.5)

            state, reward, done, info = env.step(action)

            total_reward += reward

        agent.end_episode(state)

        test_summary_writer.add_scalar(tag="Test/Episode Reward",
                                       scalar_value=total_reward,
                                       global_step=episode + 1)
        test_summary_writer.add_scalar(tag="Test/Steps to collect all Fruits",
                                       scalar_value=steps + 1,
                                       global_step=episode + 1)
    print('Completed testing in {:.03f} sec'.format(time.time() - start_time))

    env.close()
