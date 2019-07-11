from abp.examples.pysc2.tug_of_war.utilities import wave
from abp.examples.pysc2.tug_of_war.utilities import actions
 
class ActionGroup():
    def __init__(self,  array_of_actions):
        self.actions                = array_of_actions
        self.action_group_size      = len(array_of_actions)
        self.action_group_move_set  = self.get_move_set()

    def get_move_string(self):
        move = ""
        for a in self.actions:
            move += "Top: "
            move += str(a.top_self_marine) + ", "
            move += str(a.top_self_baneling) + ", "
            move += str(a.top_self_immortal) + " Bottom: "
            move += str(a.bottom_self_marine) + ", "
            move += str(a.bottom_self_baneling) + ", "
            move += str(a.bottom_self_immortal) + "| "
        return move

    def get_move_set(self):
        move_set = {}
        current_move = self.get_move_string()
        if current_move in move_set:
            move_set[current_move] += 1
        else:
            move_set.update({current_move : 1})

        return move_set.copy()
            