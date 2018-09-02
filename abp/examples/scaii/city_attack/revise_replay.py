# python -m abp.trainer.task_runner -t abp.examples.scaii.city_attack.revise_replay -f ./tasks/sky-rts/city-attack/hra/bad -v --eval

import operator
import logging

import time

from scaii.env.sky_rts.env.scenarios.city_attack import CityAttack, CityState
from scaii.env.explanation import Explanation as SkyExplanation, BarChart, BarGroup, Bar
from scaii.util import ReplayFixHelper

from abp import HRAAdaptive
from abp.explanations import Saliency

logger = logging.getLogger('root')
logger.setLevel(logging.INFO)

from enum import IntEnum

def run_task(evaluation_config, network_config, reinforce_config):
    '''
    We just want to initialize the env for action stuff
    '''
    env = CityAttack()
    # env = CityAttack("city_attack_static/attack_enemy")

    reward_types = sorted(env.reward_types())
    decomposed_rewards = {}

    for type in reward_types:
        decomposed_rewards[type] = 0

    state = env.reset()

    actions = env.actions()['actions']
    actions = sorted(actions.items(), key=operator.itemgetter(1))
    choice_descriptions = list(map(lambda x: x[0], actions))
    choices = list(map(lambda x: x[1], actions))

    # Configure network for reward type
    networks = []
    for reward_type in reward_types:
        name = reward_type
        layers = [{"type": "FC", "neurons": 50}]
        networks.append({"name": name, "layers": layers})

    network_config.networks = networks

    choose_tower = HRAAdaptive(name="tower",
                               choices=choices,
                               reward_types=reward_types,
                               network_config=network_config,
                               reinforce_config=reinforce_config)

    choose_tower.disable_learning()

    layer_names = ["HP", "Tank", "Size",
                       "City/Fort", "Friend/Enemy"]
    
    replay_fix = ReplayFixHelper(CityState)
    step = 0
    while(True):
        state = replay_fix.next()
        if state == None:
            break
        step += 1

        saliency_explanation = Saliency(choose_tower)
        explanation = SkyExplanation("Tower Capture", (40, 40))

        (tower_to_kill,
             q_values,
             combined_q_values) = choose_tower.predict(state.state.flatten())
            
        print(combined_q_values)

        q_values = q_values.data.numpy()
        combined_q_values = combined_q_values.data.numpy()
        saliencies = saliency_explanation.generate_saliencies(
            step, state.state.flatten(),
            choice_descriptions,
            ["HP", "Tank", "Small Bases", "Big Bases",
                    "Big Cities", "Small Cities", "Friend", "Enemy"],
            reshape=state.state.shape)

        decomposed_q_chart = BarChart(
            "Q Values", "Actions", "QVal By Reward Type")
        q_vals = {}
        for choice_idx, choice in enumerate(choices):
            key = choice_descriptions[choice_idx]
            group = BarGroup("Attack {}".format(key), saliency_key=key)
            explanation.add_layers(
                layer_names, saliencies[choice]["all"], key)
            q_vals[key] = combined_q_values[choice_idx]

            for reward_index, reward_type in enumerate(reward_types):
                key = "{}_{}".format(choice, reward_type)
                bar = Bar(
                    reward_type, q_values[reward_index][choice_idx], saliency_key=key)
                group.add_bar(bar)
                explanation.add_layers(
                    layer_names, saliencies[choice][reward_type], key=key)


            decomposed_q_chart.add_bar_group(group)

        explanation.with_bar_chart(decomposed_q_chart)

        action = env.new_action()
        action.attack_quadrant(tower_to_kill)
        action.skip = False
        
        replay_fix.revise_action(action, explanation)


    
    