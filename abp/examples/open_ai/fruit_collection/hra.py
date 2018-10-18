import gym
import time

from abp import HRAAdaptive
from abp.utils import clear_summary_path
from abp.explanations import PDX
from tensorboardX import SummaryWriter

# TODO: Remove action-on-import
from abp.openai import envs

def run_task(evaluation_config, network_config, reinforce_config):
    start_time = time.time()
    print('Initializing FruitCollection environment...')
    env = gym.make(evaluation_config.env)
    state = env.reset(state_representation="linear")
    LEFT, RIGHT, UP, DOWN = [0, 1, 2, 3]
    choices = [LEFT, RIGHT, UP, DOWN]
    pdx_explanation = PDX()
    print('Initialized FruitCollection environment in {:.03f} sec'.format(time.time() - start_time))

    reward_types = env.reward_types

    start_time = time.time()
    print('Initializing HRAAdaptive agent...')
    agent = HRAAdaptive(name="FruitCollecter",
                        choices=choices,
                        reward_types=reward_types,
                        network_config=network_config,
                        reinforce_config=reinforce_config)

    training_summaries_path = evaluation_config.summaries_path + "/train"
    clear_summary_path(training_summaries_path)
    train_summary_writer = SummaryWriter(training_summaries_path)

    test_summaries_path = evaluation_config.summaries_path + "/test"
    clear_summary_path(test_summaries_path)
    test_summary_writer = SummaryWriter(test_summaries_path)
    print('Initialized agent in {:.03f} sec'.format(time.time() - start_time))

    # Training Episodes
    start_time = time.time()
    print('Training for {} episodes...'.format(evaluation_config.training_episodes))
    for episode in range(evaluation_config.training_episodes):
        state = env.reset(state_representation="linear")
        total_reward = 0
        done = False
        steps = 0
        while not done:
            steps += 1
            action, q_values, combined_q_values = agent.predict(state)
            state, rewards, done, info = env.step(action, decompose_reward=True)

            for reward_type in rewards.keys():
                agent.reward(reward_type, rewards[reward_type])

            total_reward += sum(rewards.values())

        agent.end_episode(state)
        test_summary_writer.add_scalar(tag="Train/Episode Reward", scalar_value=total_reward,
                                       global_step=episode + 1)

        train_summary_writer.add_scalar(tag="Train/Episode Steps", scalar_value=steps + 1,
                                        global_step=episode + 1)
    print('Completed training in {:.03f} sec...'.format(time.time() - start_time))
    agent.disable_learning()

    # Test Episodes
    start_time = time.time()
    print('Training for {} episodes...'.format(evaluation_config.training_episodes))
    for episode in range(evaluation_config.test_episodes):
        state = env.reset(state_representation="linear")
        total_reward = 0
        done = False
        steps = 0

        while not done:
            steps += 1
            action, q_values, combined_q_values = agent.predict(state)

            if evaluation_config.render:
                env.render()
                pdx_explanation.render_decomposed_rewards(
                    action,
                    combined_q_values.data.numpy(),
                    q_values.data.numpy(),
                    env.action_names,
                    env.reward_types)

                pdx_explanation.render_all_pdx(
                    action, env.action_space,
                    q_values.data,
                    env.action_names,
                    env.reward_types)

                time.sleep(evaluation_config.sleep)

            state, reward, done, info = env.step(action)

            total_reward += reward

        agent.end_episode(state)

        test_summary_writer.add_scalar(tag="Test/Episode Reward", scalar_value=total_reward,
                                       global_step=episode + 1)
        test_summary_writer.add_scalar(tag="Test/Episode Steps", scalar_value=steps + 1,
                                       global_step=episode + 1)
    print('Completed testing in {:.03f} sec...'.format(time.time() - start_time))
    env.close()
