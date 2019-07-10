from abp.examples.pysc2.tug_of_war.utilities import wave
from abp.examples.pysc2.tug_of_war.utilities import actions
from abp.examples.pysc2.tug_of_war.utilities import action_group

import copy
 
class Episode():
    def __init__(self, episode_waves, action_group_size):         
        if (not isinstance(episode_waves, list)):
            raise ValueError("Error: the first argument is not a list for Episode(episode_waves, wave_group_size).")
        
        if (not isinstance(episode_waves[0], wave.Wave)):
            raise ValueError("Error: the first argument is not a list of waves for Episode(episode_waves, wave_group_size).")
        
        if (not isinstance(action_group_size, int)):
            raise ValueError("Error: the second argument is not an int for Episode(episode_waves, wave_group_size).")
        
        if (len(episode_waves) < action_group_size):
            raise ValueError("Error: the first argument needs to be larger than the second for Episode(episode_waves, wave_group_size).")
        
        if (len(episode_waves) <= 0):
            raise ValueError("Error: the first argument needs to be larger than 0 for Episode(episode_waves, wave_group_size).")
        
        if ( action_group_size <= 0):
            raise ValueError("Error: the second argument needs to be larger than 0 for Episode(episode_waves, wave_group_size).")
        

        self.episode_waves          = episode_waves
        self.episode_size           = len(episode_waves)
        self.action_group_size        = action_group_size
        self.list_of_action_groups    = []
        self.extract_action_groups()

    
    def extract_action_groups(self):
        empty_wave = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        i = 0
        a_group_actions = []
        while i < self.episode_size:
            if i == 0:
                a_group_actions.append(actions.Actions(wave.Wave(empty_wave), self.episode_waves[i]))                
            else:
                a_group_actions.append(actions.Actions(self.episode_waves[i-1], self.episode_waves[i]))
            if len(a_group_actions) == self.action_group_size:
                g = action_group.ActionGroup(a_group_actions)
                self.list_of_action_groups.append(g)
                a_group_actions = []
            i += 1


    def get_action_groups(self):
        return self.list_of_action_groups.copy()


    def get_action_group_count(self):
        return len(self.list_of_action_groups)
    

    def get_last_episode_wave(self):
        return copy.deepcopy(self.episode_waves[self.episode_size-1])
    

    def get_start_episode_wave(self):
        return copy.deepcopy(self.episode_waves[0])
    

    def is_player_1_winner(self):
        last_wave = self.get_last_episode_wave()
        return last_wave.is_p1_win()


    def get_p1_last_wave_building_total(self):
        last_wave = self.get_last_episode_wave()
        return last_wave.get_p1_building_totals()


    def get_p2_last_wave_building_total(self):
        last_wave = self.get_last_episode_wave()
        return last_wave.get_p2_building_totals()




    
    
