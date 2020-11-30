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