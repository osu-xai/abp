import sys
import unittest

sys.path.append('../../../abp/examples/pysc2/tug_of_war/')

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
mineral = 100
bst_mar = 101
bst_ban = 102
bst_imm = 103
            
bsb_mar = 104
bsb_ban = 105
bsb_imm = 106
pylon_self = 107
bet_mar = 108
bet_ban = 109
bet_imm = 110
beb_mar = 111
beb_ban = 112
beb_imm = 113
pylon_enemy = 114
ust_mar = 115
ust_ban = 116
ust_imm = 117
usb_mar = 118
usb_ban = 119
usb_imm = 120
ust_mar = 121
ust_ban = 122
ust_imm = 123
usb_mar  = 124
usb_ban = 125
usb_imm = 126
nex_self_top = 127
nex_self_bot = 128
nex_enemy_top = 129
nex_enemy_bot = 130

def get_waves(amount_of_waves):
    list_of_waves = []
    for i in range(amount_of_waves):

        w = [mineral + i * 100,
            bst_mar + i * 100,bst_ban + i * 100,bst_imm + i * 100,
            bsb_mar + i * 100,bsb_ban + i * 100,bsb_imm + i * 100,
            pylon_self + i * 100,
            bet_mar + i * 100,bet_ban + i * 100,bet_imm + i * 100,
            beb_mar + i * 100,beb_ban + i * 100,beb_imm + i * 100,
            pylon_enemy + i * 100,

            ust_mar + i * 100,ust_ban,ust_imm + i * 100,
            usb_mar + i * 100,usb_ban,usb_imm + i * 100,

            ust_mar + i * 100,ust_ban,ust_imm + i * 100,
            usb_mar + i * 100,usb_ban,usb_imm + i * 100,
            nex_self_top + i * 100,
            nex_self_bot + i * 100,
            nex_enemy_top + i * 100,
            nex_enemy_bot + i * 100
            ]
        w = wave.Wave(w)

        list_of_waves.append(w)
    return list_of_waves




class TestWave(unittest.TestCase):

    def test_wave(self):
        w = get_waves(1)[0]
        self.assertEqual(w.economy.self_mineral, mineral)

        self.assertEqual(w.buildings_self_top.marine, bst_mar)
        self.assertEqual(w.buildings_self_top.baneling, bst_ban)
        self.assertEqual(w.buildings_self_top.immortal, bst_imm)

        self.assertEqual(w.buildings_self_bottom.marine, bsb_mar)
        self.assertEqual(w.buildings_self_bottom.baneling, bsb_ban)
        self.assertEqual(w.buildings_self_bottom.immortal, bsb_imm)

        self.assertEqual(w.economy.self_pylon, pylon_self)

        self.assertEqual(w.buildings_enemy_top.marine, bet_mar)
        self.assertEqual(w.buildings_enemy_top.baneling, bet_ban)
        self.assertEqual(w.buildings_enemy_top.immortal, bet_imm)

        self.assertEqual(w.buildings_enemy_bottom.marine, beb_mar)
        self.assertEqual(w.buildings_enemy_bottom.baneling, beb_ban)
        self.assertEqual(w.buildings_enemy_bottom.immortal, beb_imm)


        self.assertEqual(w.economy.enemy_pylon, pylon_enemy)


        self.assertEqual(w.units_self_top.marine, ust_mar)
        self.assertEqual(w.units_self_top.baneling, ust_ban)
        self.assertEqual(w.units_self_top.immortal, ust_imm)

        self.assertEqual(w.units_self_bottom.marine, usb_mar)
        self.assertEqual(w.units_self_bottom.baneling, usb_ban)
        self.assertEqual(w.units_self_bottom.immortal, usb_imm)


        self.assertEqual(w.units_self_top.marine, ust_mar)
        self.assertEqual(w.units_self_top.baneling, ust_ban)
        self.assertEqual(w.units_self_top.immortal, ust_imm)

        self.assertEqual(w.units_self_bottom.marine, usb_mar)
        self.assertEqual(w.units_self_bottom.baneling, usb_ban)
        self.assertEqual(w.units_self_bottom.immortal, usb_imm)

        self.assertEqual(w.nexus_self_top, nex_self_top)
        self.assertEqual(w.nexus_self_bottom, nex_self_bot)
        self.assertEqual(w.nexus_enemy_top, nex_enemy_top)
        self.assertEqual(w.nexus_enemy_bottom, nex_enemy_bot)

# top/bottom oriented tests


        self.assertEqual(w.top.buildings_self.marine, bst_mar)
        self.assertEqual(w.top.buildings_self.baneling, bst_ban)
        self.assertEqual(w.top.buildings_self.immortal, bst_imm)

        self.assertEqual(w.bottom.buildings_self.marine, bsb_mar)
        self.assertEqual(w.bottom.buildings_self.baneling, bsb_ban)
        self.assertEqual(w.bottom.buildings_self.immortal, bsb_imm)


        self.assertEqual(w.top.buildings_enemy.marine, bet_mar)
        self.assertEqual(w.top.buildings_enemy.baneling, bet_ban)
        self.assertEqual(w.top.buildings_enemy.immortal, bet_imm)

        self.assertEqual(w.bottom.buildings_enemy.marine, beb_mar)
        self.assertEqual(w.bottom.buildings_enemy.baneling, beb_ban)
        self.assertEqual(w.bottom.buildings_enemy.immortal, beb_imm)


        self.assertEqual(w.top.units_self.marine, ust_mar)
        self.assertEqual(w.top.units_self.baneling, ust_ban)
        self.assertEqual(w.top.units_self.immortal, ust_imm)

        self.assertEqual(w.bottom.units_self.marine, usb_mar)
        self.assertEqual(w.bottom.units_self.baneling, usb_ban)
        self.assertEqual(w.bottom.units_self.immortal, usb_imm)


        self.assertEqual(w.top.units_self.marine, ust_mar)
        self.assertEqual(w.top.units_self.baneling, ust_ban)
        self.assertEqual(w.top.units_self.immortal, ust_imm)

        self.assertEqual(w.bottom.units_self.marine, usb_mar)
        self.assertEqual(w.bottom.units_self.baneling, usb_ban)
        self.assertEqual(w.bottom.units_self.immortal, usb_imm)

        self.assertEqual(w.top.nexus_self, nex_self_top)
        self.assertEqual(w.bottom.nexus_self, nex_self_bot)
        self.assertEqual(w.top.nexus_enemy, nex_enemy_top)
        self.assertEqual(w.bottom.nexus_enemy, nex_enemy_bot)

if __name__ == "__main__":
    unittest.main()
