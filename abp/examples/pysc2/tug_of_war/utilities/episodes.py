from abp.examples.pysc2.tug_of_war.utilities import episode
from abp.examples.pysc2.tug_of_war.utilities import wave
import copy
from collections import Counter
 
class Episodes():
    def __init__(self, raw_data, raw_data_size, wave_group_size):
        self.raw_data           = raw_data
        self.raw_data_size      = raw_data_size
        self.wave_group_size    = wave_group_size
        self.episodes           = self.seperate_episodes(self.raw_data, self.raw_data_size)

    def convert_to_wave(self, raw_data, raw_data_size):
        wave_list = []
        for d in range(raw_data_size):
            w = wave.Wave(raw_data[d][0]) #0 index is to get just the input array we dont need output
            wave_list.append(w)

        return wave_list


    def seperate_episodes(self, raw_data, raw_data_size):
        wave_list = self.convert_to_wave(self.raw_data, self.raw_data_size)
        episode_waves = []
        list_of_episodes = []
        for w in range(1, len(wave_list)):
            prev_w = wave_list[w-1]
            curr_w = wave_list[w]
            if(curr_w.is_reset(prev_w, curr_w)):
                try:
                    list_of_episodes.append(episode.Episode(episode_waves, self.wave_group_size))
                    episode_waves = []
                except ValueError:
                    print("ommiting this wave...")
                    episode_waves = []
            else:
                episode_waves.append(wave_list[w])

        return list_of_episodes.copy()

    def get_episode_count(self):
        return len(self.episodes)

    def get_episode(self, episode_index):
        return copy.deepcopy(self.episodes[episode_index])

    def get_last_episode(self):
        return copy.deepcopy(self.episodes[len(self.episodes)-1])

    def get_total_p1_wins(self):
        p1_wins = 0
        for episode in self.episodes:
            if episode.is_player_1_winner():
                p1_wins += 1
        return p1_wins
    
    def get_p1_buildings(self):
        p1_buildings = [0,0,0, 0,0,0] # loss: marine, baneling, immortal Win: marine, baneling, immortal 

        for episode in self.episodes:
            if not episode.is_player_1_winner():
                p1_buildings[0] += episode.get_last_episode_wave().buildings_self_top.marine
                p1_buildings[1] += episode.get_last_episode_wave().buildings_self_top.baneling
                p1_buildings[2] += episode.get_last_episode_wave().buildings_self_top.immortal
                p1_buildings[0] += episode.get_last_episode_wave().buildings_self_bottom.marine
                p1_buildings[1] += episode.get_last_episode_wave().buildings_self_bottom.baneling
                p1_buildings[2] += episode.get_last_episode_wave().buildings_self_bottom.immortal
            else:
                p1_buildings[3] += episode.get_last_episode_wave().buildings_self_top.marine
                p1_buildings[4] += episode.get_last_episode_wave().buildings_self_top.baneling
                p1_buildings[5] += episode.get_last_episode_wave().buildings_self_top.immortal
                p1_buildings[3] += episode.get_last_episode_wave().buildings_self_bottom.marine
                p1_buildings[4] += episode.get_last_episode_wave().buildings_self_bottom.baneling
                p1_buildings[5] += episode.get_last_episode_wave().buildings_self_bottom.immortal
        return p1_buildings


    def get_p2_buildings(self):
        p2_buildings = [0,0,0, 0,0,0] # loss: marine, baneling, immortal Win: marine baneling, immortal 

        for episode in self.episodes:
            if episode.is_player_1_winner():
                p2_buildings[0] += episode.get_last_episode_wave().buildings_enemy_top.marine
                p2_buildings[1] += episode.get_last_episode_wave().buildings_enemy_top.baneling
                p2_buildings[2] += episode.get_last_episode_wave().buildings_enemy_top.immortal
                p2_buildings[0] += episode.get_last_episode_wave().buildings_enemy_bottom.marine
                p2_buildings[1] += episode.get_last_episode_wave().buildings_enemy_bottom.baneling
                p2_buildings[2] += episode.get_last_episode_wave().buildings_enemy_bottom.immortal
            else:
                p2_buildings[3] += episode.get_last_episode_wave().buildings_enemy_top.marine
                p2_buildings[4] += episode.get_last_episode_wave().buildings_enemy_top.baneling
                p2_buildings[5] += episode.get_last_episode_wave().buildings_enemy_top.immortal
                p2_buildings[3] += episode.get_last_episode_wave().buildings_enemy_bottom.marine
                p2_buildings[4] += episode.get_last_episode_wave().buildings_enemy_bottom.baneling
                p2_buildings[5] += episode.get_last_episode_wave().buildings_enemy_bottom.immortal
        return p2_buildings

    def get_win_loss_sequence(self):
        win_loss_sequence = [0]
        count = 0
        for episode in self.episodes:
            last_wave = episode.get_last_episode_wave()
            if last_wave.is_p1_win():
                win_loss_sequence.append(win_loss_sequence[count]+1)
            else:
                win_loss_sequence.append(win_loss_sequence[count]-1)
            count += 1
        return win_loss_sequence.copy()

    def merge_dicts(self, dict1, dict2):
        merged_dict = {x: dict1.get(x, 0) + dict2.get(x, 0) 
                    for x in set(dict1).union(dict2)} 
        return merged_dict.copy()
    
    def get_binned_move_sets(self):
        move_sets = []
        for d in range(int(40/self.wave_group_size)):
            move_sets.append({})
        for i in range(len(self.episodes)):
            for j in range(len(self.episodes[i].list_of_action_groups)):
                curr_move_set = self.episodes[i].list_of_action_groups[j].get_move_set()
                temp_move_set = move_sets[j].copy()

                move_sets[j] = self.merge_dicts(curr_move_set, temp_move_set)

        return move_sets.copy()
                    
    def get_end_building_frequencies(self):
        marine_dict = {}
        baneling_dict = {}
        immortal_dict = {}
        print(len(self.episodes))
        for episode in self.episodes:
            last_wave = episode.get_last_episode_wave()

            tms = "Top Player 1 Marine (" + str(last_wave.top.buildings_self.marine) + ")"
            tbs = "Top Player 1 Baneling (" + str(last_wave.top.buildings_self.baneling) + ")"
            tis = "Top Player 1 Immortal (" + str(last_wave.top.buildings_self.immortal) + ")"
            bms = "Bottom Player 1 Marine (" + str(last_wave.bottom.buildings_self.marine) + ")"
            bbs = "Bottom Player 1 Baneling (" + str(last_wave.bottom.buildings_self.baneling) + ")"
            bis = "Bottom Player 1 Immortal (" + str(last_wave.bottom.buildings_self.immortal) + ")"

            tme = "Top Player 2 Marine (" + str(last_wave.top.buildings_enemy.marine) + ")"
            tbe = "Top Player 2 Baneling (" + str(last_wave.top.buildings_enemy.baneling) + ")"
            tie = "Top Player 2 Immortal (" + str(last_wave.top.buildings_enemy.immortal) + ")"
            bme = "Bottom Player 2 Marine (" + str(last_wave.bottom.buildings_enemy.marine) + ")"
            bbe = "Bottom Player 2 Baneling (" + str(last_wave.bottom.buildings_enemy.baneling) + ")"
            bie = "Bottom Player 2 Immortal (" + str(last_wave.bottom.buildings_enemy.immortal) + ")"

            if tms in marine_dict:
                marine_dict[tms] += 1
            else:
                marine_dict.update({tms : 1})

            if tbs in baneling_dict:
                baneling_dict[tbs] += 1
            else:
                baneling_dict.update({tbs : 1})
            
            if tis in immortal_dict:
                immortal_dict[tis] += 1
            else:
                immortal_dict.update({tis : 1})

            if bms in marine_dict:
                marine_dict[bms] += 1
            else:
                marine_dict.update({bms : 1})

            if bbs in baneling_dict:
                baneling_dict[bbs] += 1
            else:
                baneling_dict.update({bbs : 1})
            
            if bis in immortal_dict:
                immortal_dict[bis] += 1
            else:
                immortal_dict.update({bis : 1})

            
            
            if tme in marine_dict:
                marine_dict[tme] += 1
            else:
                marine_dict.update({tme : 1})

            if tbe in baneling_dict:
                baneling_dict[tbe] += 1
            else:
                baneling_dict.update({tbe : 1})
            
            if tie in immortal_dict:
                immortal_dict[tie] += 1
            else:
                immortal_dict.update({tie : 1})

            if bme in marine_dict:
                marine_dict[bme] += 1
            else:
                marine_dict.update({bme : 1})

            if bbe in baneling_dict:
                baneling_dict[bbe] += 1
            else:
                baneling_dict.update({bbe : 1})
            
            if bie in immortal_dict:
                immortal_dict[bie] += 1
            else:
                immortal_dict.update({bie : 1})

        return marine_dict, baneling_dict, immortal_dict