import numpy as np

def GFVs_all_1(state, is_max_steps):
    # eval 2 features
    current_idx = 0
    features = np.zeros(network_config.shared_layers)
#         norm_vector = np.ones(5)
    norm_vector = np.array([1500, 100, 30, 200, 5000])
#         print(current_idx)
    # self-mineral feature idx 8, enemy-mineral idx 9. self-pylone idx 7, enemy-pylon 14
    features[current_idx] = (state[7] * 75 + 100) / norm_vector[0]
    current_idx += 1
    features[current_idx] = (state[14] * 75 + 100) / norm_vector[0]
    current_idx += 1
#         print(current_idx)

    # features idx: The number of self-units will be spawned, length : 6, The number of enemy-units will be spawned, length : 6
    # state idx: The number of self-building 1-6, The number of self-building 8-13, 
    features[current_idx : current_idx + 6] = state[1:7] / norm_vector[1]
    current_idx += 6
    features[current_idx : current_idx + 6] = state[8:14] / norm_vector[1]
    current_idx += 6
#         print(current_idx)

    # features idx: The accumulative number of each type of unit in each range from now to the end of the game: length : 30, for enemy: length : 30
    agent_attacking_units, enemy_attacking_units = env.get_attacking()
    features[current_idx : current_idx + 12] = np.array(state[15:27]) / norm_vector[2]
    current_idx += 12
    features[current_idx : current_idx + 3] = agent_attacking_units[:3] / norm_vector[2]
    current_idx += 3
    features[current_idx : current_idx + 12] = np.array(state[27:39]) / norm_vector[2]
    current_idx += 12
    features[current_idx : current_idx + 3] = agent_attacking_units[3:] / norm_vector[2]
    current_idx += 3
#         print(current_idx)

    features[current_idx : current_idx + 3] = enemy_attacking_units[:3] / norm_vector[2]
    current_idx += 3
    features[current_idx : current_idx + 12] = np.array(state[39:51]) / norm_vector[2]
    current_idx += 12
    features[current_idx : current_idx + 3] = enemy_attacking_units[3:] / norm_vector[2]
    current_idx += 3
    features[current_idx : current_idx + 12] = np.array(state[51:63]) / norm_vector[2]
    current_idx += 12
#         print(current_idx)

    
    damage_to_nexus, get_damage_nexus = env.get_damage_to_nexus()
    features[current_idx : current_idx + 6] = np.array(damage_to_nexus) / norm_vector[3]
    current_idx += 6
    features[current_idx : current_idx + 6] = np.array(get_damage_nexus) / norm_vector[3]
    current_idx += 6
#         print(current_idx)

    # features idx: The number of which friendly troops kills which enemy troops: length : 18, for enemy: length : 18
    unit_kills, unit_be_killed = env.get_unit_kill()
    features[current_idx : current_idx + 18] = np.array(unit_kills) / norm_vector[4]
    current_idx += 18
    features[current_idx : current_idx + 18] = np.array(unit_be_killed) / norm_vector[4]
    current_idx += 18
#         print(current_idx)

    return features

def reward_vector(state, done):
    
    hp_vector = np.array([state[63], state[64], state[65], state[66]])
    min_value, idx = np.min(hp_vector), np.argmin(hp_vector)
    reward_vector = [0] * 4
    
    if done:
        if min_value == 2000:
            reward_vector = [0.25] * 4
        else:
            reward_vector[idx] = 1

    return reward_vector

def self_unit_spawn(state):
    return np.array(state[1:7])

# features idx: Damage of each friendly unit to each Nexus: length : 6, for enemy: length : 6
def self_damage_to_nexus(env):
    damage_to_nexus, _ = env.get_damage_to_nexus()
    return np.array(damage_to_nexus) / 2000

def self_get_damage_nexus(env):
    _, get_damage_nexus = env.get_damage_to_nexus()
    return np.array(get_damage_nexus) / 2000

def self_create_building(action):
    return action

def get_features(features_list, state, previous_action, env, done):
    features = []
    for f in features_list:
        if f == "reward_vector":
            features.append(reward_vector(state, done))
        if f == "self_unit_spawn":
            features.append(self_unit_spawn(state))
        if f == "self_damage_to_nexus":
            features.append(self_damage_to_nexus(env))
        if f == "self_get_damage_nexus":
            features.append(self_get_damage_nexus(env))
        if f == "self_create_building":
            features.append(self_create_building(previous_action))
    return np.concatenate(features)