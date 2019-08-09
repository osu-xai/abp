class Node():
    def __init__(self, name, state = None, reward = 0, parent = None, q_value_after_state = 0, parent_action = None):
        self.parent = parent
        self.q_value_after_state = 0
        self.best_q_value = 0
#         self.reward = reward
#         self.accumulate_reward = 0
        self.children = []
        self.action_dict = {}
        self.actions = []
        self.state = state
        self.name = name
        self.best_child = None
        self.best_action = None
        self.parent_action = parent_action
        
    def add_child(self, sub_node, action):
        self.children.append(sub_node)
        self.action_dict[str(action)] = sub_node
        
    def save(self):
        pass
    
    def sort_children(self):
        pass
    # Give the node find the worst reward sibling
    # add child (sort by reward)
    # get best child
    # get worst child
    # save tree 
    # action dictionary corresponding to [action1][action2]