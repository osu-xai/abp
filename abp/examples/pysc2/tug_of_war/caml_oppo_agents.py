import numpy as np

def immortal_agent(state):
    if state[3] <= state[6]:
        # Top lane
        actions = [[0, 0, 10, 0, 0, 0, 0]]
    else:
        # Bottom lane
        actions = [[0, 0, 0, 0, 0, 10, 0]]
    
    return actions, 0

def baneling_agent(state):
    if state[-1] % 2 == 0:
        # Top lane
        actions = [[0, 10, 0, 0, 0, 0, 0]]
    else:
        # Bottom lane
        actions = [[0, 0, 0, 0, 10, 0, 0]]
    return actions, 0

def marine_agent(state):
    if state[-1] % 2 == 0:
        # Top lane
        actions = [[10, 0, 0, 0, 0, 0, 0]]
    else:
        # Bottom lane
        actions = [[0, 0, 0, 10, 0, 0, 0]]
    
    return actions, 0

def fine_immortal_agent(state):
    oppo_top_lane_num = sum(state[8:11])
    oppo_bottom_lane_num = sum(state[11:14])
    
    if oppo_top_lane_num >= oppo_bottom_lane_num:
        actions = [[0, 0, 10, 0, 0, 0, 0]]
    else:
        # Bottom lane
        actions = [[0, 0, 0, 0, 0, 10, 0]]
    
    return actions, 0

def fine_baneling_agent(state):
    oppo_top_lane_num = sum(state[8:11])
    oppo_bottom_lane_num = sum(state[11:14])
    
    if oppo_top_lane_num >= oppo_bottom_lane_num:
        # Top lane
        actions = [[0, 10, 0, 0, 0, 0, 0]]
    else:
        # Bottom lane
        actions = [[0, 0, 0, 0, 10, 0, 0]]
    return actions, 0

def fine_marine_agent(state):
    oppo_top_lane_num = sum(state[8:11])
    oppo_bottom_lane_num = sum(state[11:14])
    
    if oppo_top_lane_num >= oppo_bottom_lane_num:
        actions = [[10, 0, 0, 0, 0, 0, 0]]
    else:
        # Bottom lane
        actions = [[0, 0, 0, 10, 0, 0, 0]]
    
    return actions, 0

def smart_immortal_agent(state):
    oppo_top_lane_diff = sum(np.array(state[8:11]) * np.array([50, 75, 200])) - sum(np.array(state[1:4]) * np.array([50, 75, 200]))
    oppo_bottom_lane_diff = sum(np.array(state[11:14]) * np.array([50, 75, 200])) - sum(np.array(state[4:7]) * np.array([50, 75, 200]))
    
    if oppo_top_lane_diff >= oppo_bottom_lane_diff:
        actions = [[0, 0, 10, 0, 0, 0, 0]]
    else:
        # Bottom lane
        actions = [[0, 0, 0, 0, 0, 10, 0]]
    
    return actions, 0

def smart_baneling_agent(state):
    oppo_top_lane_diff = sum(np.array(state[8:11]) * np.array([50, 75, 200])) - sum(np.array(state[1:4]) * np.array([50, 75, 200]))
    oppo_bottom_lane_diff = sum(np.array(state[11:14]) * np.array([50, 75, 200])) - sum(np.array(state[4:7]) * np.array([50, 75, 200]))
    
    if oppo_top_lane_diff >= oppo_bottom_lane_diff:
        # Top lane
        actions = [[0, 10, 0, 0, 0, 0, 0]]
    else:
        # Bottom lane
        actions = [[0, 0, 0, 0, 10, 0, 0]]
    return actions, 0

def smart_marine_agent(state):
    oppo_top_lane_diff = sum(np.array(state[8:11]) * np.array([50, 75, 200])) - sum(np.array(state[1:4]) * np.array([50, 75, 200]))
    oppo_bottom_lane_diff = sum(np.array(state[11:14]) * np.array([50, 75, 200])) - sum(np.array(state[4:7]) * np.array([50, 75, 200]))
    
    if oppo_top_lane_diff >= oppo_bottom_lane_diff:
        actions = [[10, 0, 0, 0, 0, 0, 0]]
    else:
        # Bottom lane
        actions = [[0, 0, 0, 10, 0, 0, 0]]
    
    return actions, 0

def map_to_agent(agent_names):
    agents = []
    for an in agent_names:
        if an == "marine":
            agents.append(marine_agent)
        elif an == "baneling":
            agents.append(baneling_agent)
        elif an == "immortal":
            agents.append(immortal_agent)
            
        elif an == "smart_marine":
            agents.append(smart_marine_agent)
        elif an == "smart_baneling":
            agents.append(smart_baneling_agent)
        elif an == "smart_immortal":
            agents.append(smart_immortal_agent)
            
        elif an == "fine_marine":
            agents.append(fine_marine_agent)
        elif an == "fine_baneling":
            agents.append(fine_baneling_agent)
        elif an == "fine_immortal":
            agents.append(fine_immortal_agent)
        else:
            print("agent match error!!")
            print("agent name: {}".format(an))
            input()
            
    return agents