class node():
    def __init__(self, name, state, reward = None, parent = None, total_reward = 0):
        self.parent = parent
        self.reward = reward
        self.total_reward = 0
        self.children = []
        self.action_dict = {}
        self.state = state
        self.name = name
        
    def add_child(self, sub_node, action):
        self.children.append(sub_node)
        self[str(action)] = sub_node
        
    def save(self):
        pass
    # Give the node find the worst reward sibling
    # add child (sort by reward)
    # get best child
    # get worst child
    # save tree 
    # action dictionary 