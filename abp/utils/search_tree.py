class node():
    def __init__(self, name, reward = None, parent = None, total_reward = 0):
        self.parent = parent
        self.reward = reward
        self.total_reward = 0
        self.children = []
        self.name = name
        
    def add_child(sub_node):
        self.children.append(sub_node)
    
    # Give the node find the worst reward sibling
    # add child (sort by reward)
    # get best child
    # get worst child
    # save tree 
    # action dictionary 