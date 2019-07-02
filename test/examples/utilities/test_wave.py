import sys
import unittest

from abp.examples.pysc2.tug_of_war import wave


'''

1    mineral, 
3    building self top, 
3    building self bottom, 
1    pylon self
3    building enemy top, 
3    building enemy bottom, 
1    pylon enemy 
3    unit self top, 
3    unit self bottom, 
3    unit enemy top, 
3    unit enemy bottom, 
1    nexus health self top
1    nexus health self bottom
1    nexus health enemy top
1    nexus health enemy bottom
'''
wave1 = [
            ['min',
            'bst_mar','bst_ban','bst_imm',
            'bsb_mar','bsb_ban','bsb_imm',
            'pylon_self',
            'bet_mar','bet_ban','bet_imm',
            'beb_mar','beb_ban','beb_imm',
            'pylon_enemy',

            'ust_mar','ust_ban','ust_imm',
            'usb_mar','usb_ban','usb_imm',

            'ust_mar','ust_ban','ust_imm',
            'usb_mar','usb_ban','usb_imm',
            'nex_self_top',
            'nex_self_bot',
            'nex_enemy_top',
            'nex_enemy_bot'
            ]
        ]




class TestWave(unittest.TestCase):

    def test_wave(self):
        wave = Wave(wave1)
        self.assertEqual(wave.economy.self_mineral, 'min')

        self.assertEqual(wave.building_self_top.marine, 'bst_mar')
        self.assertEqual(wave.building_self_top.baneling, 'bst_ban')
        self.assertEqual(wave.building_self_top.immortal, 'bst_imm')

        self.assertEqual(wave.building_self_bottom.marine, 'bsb_mar')
        self.assertEqual(wave.building_self_bottom.baneling, 'bsb_ban')
        self.assertEqual(wave.building_self_bottom.immortal, 'bsb_imm')

        self.assertEqual(wave.economy.self_pylon, 'pylon_self')

        self.assertEqual(wave.building_enemy_top.marine, 'bet_mar')
        self.assertEqual(wave.building_enemy_top.baneling, 'bet_ban')
        self.assertEqual(wave.building_enemy_top.immortal, 'bet_imm')

        self.assertEqual(wave.building_enemy_bottom.marine, 'beb_mar')
        self.assertEqual(wave.building_enemy_bottom.baneling, 'beb_ban')
        self.assertEqual(wave.building_enemy_bottom.immortal, 'beb_imm')


        self.assertEqual(wave.economy.enemy_pylon, 'pylon_enemy')


        self.assertEqual(wave.unit_self_top.marine, 'ust_mar')
        self.assertEqual(wave.unit_self_top.baneling, 'ust_ban')
        self.assertEqual(wave.unit_self_top.immortal, 'ust_imm')

        self.assertEqual(wave.unit_self_bottom.marine, 'usb_mar')
        self.assertEqual(wave.unit_self_bottom.baneling, 'usb_ban')
        self.assertEqual(wave.unit_self_bottom.immortal, 'usb_imm')


        self.assertEqual(wave.unit_self_top.marine, 'ust_mar')
        self.assertEqual(wave.unit_self_top.baneling, 'ust_ban')
        self.assertEqual(wave.unit_self_top.immortal, 'ust_imm')

        self.assertEqual(wave.unit_self_bottom.marine, 'usb_mar')
        self.assertEqual(wave.unit_self_bottom.baneling, 'usb_ban')
        self.assertEqual(wave.unit_self_bottom.immortal, 'usb_imm')

        self.assertEqual(wave.nexus_self_top, 'nex_self_top')
        self.assertEqual(wave.nexus_self_bottom, 'nex_self_bot')
        self.assertEqual(wave.nexus_enemy_top, 'nex_enemy_top')
        self.assertEqual(wave.nexus_enemy_bottom, 'nex_enemy_bot')



if __name__ == "__main__":
    unittest.main()
