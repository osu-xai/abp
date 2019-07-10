from abp.examples.pysc2.tug_of_war.utilities import wave
import copy


class Actions():
    def __init__(self, prev_wave, curr_wave):   
        self.prev_wave = prev_wave
        self.curr_wave = curr_wave

        
        self.top_self_marine = 0
        self.top_self_baneling = 0
        self.top_self_immortal = 0

        self.bottom_self_marine = 0
        self.bottom_self_baneling = 0
        self.bottom_self_immortal = 0

        self.top_enemy_marine = 0
        self.top_enemy_baneling = 0
        self.top_enemy_immortal = 0

        self.bottom_enemy_marine = 0
        self.bottom_enemy_baneling = 0
        self.bottom_enemy_immortal = 0

        self.actions   = self.get_actions()


    def get_actions(self):
        self.top_self_marine = self.curr_wave.top.buildings_self.marine - self.prev_wave.top.buildings_self.marine
        self.top_self_baneling = self.curr_wave.top.buildings_self.baneling - self.prev_wave.top.buildings_self.baneling
        self.top_self_immortal = self.curr_wave.top.buildings_self.immortal - self.prev_wave.top.buildings_self.immortal

        self.bottom_self_marine = self.curr_wave.bottom.buildings_self.marine - self.prev_wave.bottom.buildings_self.marine
        self.bottom_self_baneling = self.curr_wave.bottom.buildings_self.baneling - self.prev_wave.bottom.buildings_self.baneling
        self.bottom_self_immortal = self.curr_wave.bottom.buildings_self.immortal - self.prev_wave.bottom.buildings_self.immortal

        self.top_enemy_marine = self.curr_wave.top.buildings_enemy.marine - self.prev_wave.top.buildings_enemy.marine
        self.top_enemy_baneling = self.curr_wave.top.buildings_enemy.baneling - self.prev_wave.top.buildings_enemy.baneling
        self.top_enemy_immortal = self.curr_wave.top.buildings_enemy.immortal - self.prev_wave.top.buildings_enemy.immortal
        
        self.bottom_enemy_marine = self.curr_wave.bottom.buildings_enemy.marine - self.prev_wave.bottom.buildings_enemy.marine
        self.bottom_enemy_baneling = self.curr_wave.bottom.buildings_enemy.baneling - self.prev_wave.bottom.buildings_enemy.baneling
        self.bottom_enemy_immortal = self.curr_wave.bottom.buildings_enemy.immortal - self.prev_wave.bottom.buildings_enemy.immortal
        