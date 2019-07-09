from abp.examples.pysc2.tug_of_war.utilities import wave_group
from abp.examples.pysc2.tug_of_war.utilities import wave
import copy

class Episode():
    def __init__(self, episode_waves, wave_group_size):         
        if (not isinstance(episode_waves, list)):
            raise ValueError("Error: the first argument is not a list for Episode(episode_waves, wave_group_size).")
        
        if (not isinstance(episode_waves[0], wave.Wave)):
            raise ValueError("Error: the first argument is not a list of waves for Episode(episode_waves, wave_group_size).")
        
        if (not isinstance(wave_group_size, int)):
            raise ValueError("Error: the second argument is not an int for Episode(episode_waves, wave_group_size).")
        
        if (len(episode_waves) < wave_group_size):
            raise ValueError("Error: the first argument needs to be larger than the second for Episode(episode_waves, wave_group_size).")
        
        if (len(episode_waves) <= 0):
            raise ValueError("Error: the first argument needs to be larger than 0 for Episode(episode_waves, wave_group_size).")
        
        if ( wave_group_size <= 0):
            raise ValueError("Error: the second argument needs to be larger than 0 for Episode(episode_waves, wave_group_size).")
        

        self.episode_waves          = episode_waves
        self.episode_size           = len(episode_waves)
        self.wave_group_size        = wave_group_size
        self.list_of_wave_groups    = []
        self.extract_wave_groups()
    
    def extract_wave_groups(self):
        i = 0
        w_group_waves = []
        while i < self.episode_size:
            w_group_waves.append(self.episode_waves[i])
            if len(w_group_waves) == self.wave_group_size:
                g = wave_group.WaveGroup(w_group_waves)
                self.list_of_wave_groups.append(g)
                w_group_waves = []
            i += 1

    def get_wave_groups(self):
        return self.list_of_wave_groups.copy()


    def get_wave_group_count(self):
        return len(self.list_of_wave_groups)
    
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



    
    
