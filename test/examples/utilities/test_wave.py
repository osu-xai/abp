import sys
import unittest

sys.path.append('../../../abp/examples/pysc2/tug_of_war/utilities')

from abp.examples.pysc2.tug_of_war.utilities import wave



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

def get_waves_raw_data(amount_of_waves):
    list_of_waves = []
    for i in range(amount_of_waves):

        w = [mineral + i * 100,
            bst_mar + i * 100,bst_ban + i * 100,bst_imm + i * 100,
            bsb_mar + i * 100,bsb_ban + i * 100,bsb_imm + i * 100,
            pylon_self + i * 100,
            bet_mar + i * 100,bet_ban + i * 100,bet_imm + i * 100,
            beb_mar + i * 100,beb_ban + i * 100,beb_imm + i * 100,
            pylon_enemy + i * 100,

            ust_mar + i * 100,ust_ban + i * 100,ust_imm + i * 100,
            usb_mar + i * 100,usb_ban + i * 100,usb_imm + i * 100,

            ust_mar + i * 100,ust_ban + i * 100,ust_imm + i * 100,
            usb_mar + i * 100,usb_ban + i * 100,usb_imm + i * 100,
            nex_self_top + i * 100,
            nex_self_bot + i * 100,
            nex_enemy_top + i * 100,
            nex_enemy_bot + i * 100
            ]

        list_of_waves.append(w)
    return list_of_waves




def get_waves(amount_of_waves):
    list_of_waves = []
    raw_wave_data = get_waves_raw_data(amount_of_waves)
    for d in raw_wave_data:
        w = wave.Wave(d)
        list_of_waves.append(w)
    return list_of_waves

def get_t_wave(t1,t2,b1,b2):
    w = get_waves(1)[0]
    w.top.nexus_self = t1
    w.top.nexus_enemy = t2
    w.bottom.nexus_self = b1
    w.bottom.nexus_enemy = b2
    return w




class TestWave(unittest.TestCase):

    def test_is_reset(self):
        waves = get_waves(2)
        w0 = waves[0]
        w1 = waves[1]
        self.assertTrue(w0.is_reset(w1, w0))
        self.assertFalse(w0.is_reset(w0, w1))

    # checking for all relationships that will determine a win, loss, or tie
    
# is a win for p1???
    def test_is_p1_win(self):
    # t1 = t2 = b1 = b2     player 2 wins 0,0,0,0(tie)
        self.assertFalse(get_t_wave(0,0,0,0).is_p1_win())
    # t1 = t2 = b1 = b2     player 2 wins 2000,2000,2000,2000 (tie)
        self.assertFalse(get_t_wave(2000,2000,2000,2000).is_p1_win())
    # t1 > t2 > b1 > b2     player 1 wins 2000,1000,500,0
        self.assertTrue(get_t_wave(2000,1000,500,0).is_p1_win())
    # t1 > t2 > b1 > b2     player 1 wins 2000,1000,500,100
        self.assertTrue(get_t_wave(2000,1000,500,100).is_p1_win())
    # t1 < t2 < b1 < b2     player 2 wins 0,500,1000,2000
        self.assertFalse(get_t_wave(0,500,1000,2000).is_p1_win())
       
                            ## >==
    # t1 > t2 = b1 = b2     player 1 wins 2000,0,0,0
        self.assertTrue(get_t_wave(2000,0,0,0).is_p1_win())
    # t1 = t2 > b1 = b2     player 2 wins 2000,2000,0,0 (tie)
        self.assertFalse(get_t_wave(2000,2000,0,0).is_p1_win())
    # t1 = t2 = b1 > b2     player 1 wins 500,500,500,0
        self.assertTrue(get_t_wave(500,500,500,0).is_p1_win())

                            ## <==
    # t1 < t2 = b1 = b2     player 2 wins 100,2000,2000,2000
        self.assertFalse(get_t_wave(100,2000,2000,2000).is_p1_win())
    # t1 = t2 < b1 = b2     player 2 wins 500,500,2000,2000 (tie)
        self.assertFalse(get_t_wave(500,500,2000,2000).is_p1_win())
    # t1 = t2 = b1 < b2     player 2 wins 500,500 500,2000
        self.assertFalse(get_t_wave(500,500,500,2000).is_p1_win())
    # t1 = t2 = b1 < b2     player 2 wins 0,0 0,2000
        self.assertFalse(get_t_wave(0,0,0,2000).is_p1_win())
    

                            ## >>=
    # t1 > t2 > b1 = b2     player 1 wins 2000,1000,500,500
        self.assertTrue(get_t_wave(2000,1000,500,500).is_p1_win())
    # t1 > t2 > b1 = b2     player 1 wins 2000,1000,0,0
        self.assertTrue(get_t_wave(2000,1000,0,0).is_p1_win())
    # t1 = t2 > b1 > b2     player 1 wins 2000,2000,500,0
        self.assertTrue(get_t_wave(2000,2000,2000,0).is_p1_win())
    # t1 > t2 = b1 > b2     player 1 wins 2000,500,500,0
        self.assertTrue(get_t_wave(2000,500,500,0).is_p1_win())

                            ## <<=
    # t1 < t2 < b1 = b2     player 2 wins 0,500,2000,2000
        self.assertFalse(get_t_wave(0,500,2000,2000).is_p1_win())
    # t1 < t2 = b1 < b2     player 2 wins 0,1000,1000,2000
        self.assertFalse(get_t_wave(0,1000,1000,2000).is_p1_win())
    # t1 = t2 < b1 < b2     player 2 wins 0,0,500,2000
        self.assertFalse(get_t_wave(0,0,500,2000).is_p1_win())
    # t1 = t2 < b1 < b2     player 2 wins 100,100,500,2000
        self.assertFalse(get_t_wave(100,100,500,2000).is_p1_win())

                            ## >><
    # t1 > t2 > b1 < b2  (b2 == t2, b1 == 0)   player 2 wins 2000,1000,0,1000 
        self.assertFalse(get_t_wave(2000,1000,0,1000).is_p1_win())
    # t1 > t2 > b1 < b2  (b2 == t2)            player 2 wins 2000,1000,500,1000 
        self.assertFalse(get_t_wave(2000,1000,500,1000).is_p1_win())
    # t1 > t2 > b1 < b2  (b2 > t1)             player 2 wins 1500,1000,500,2000 
        self.assertFalse(get_t_wave(1500,1000,500,2000).is_p1_win())
    # t1 > t2 > b1 < b2  (t1 > b2 > t2)        player 2 wins 2000,1000,500,1800 
        self.assertFalse(get_t_wave(2000,1000,500,1800).is_p1_win())
    # t1 > t2 > b1 < b2  (b2 < t2)             player 2 wins 2000,1000,500,800 
        self.assertFalse(get_t_wave(2000,1000,500,800).is_p1_win())
    
    # t1 < t2 > b1 > b2     player 2 wins 0,2000,1000,500
        self.assertFalse(get_t_wave(0,2000,1000,500).is_p1_win())
    # t1 < t2 > b1 > b2     player 2 wins 0,2000,1000,0
        self.assertFalse(get_t_wave(0,2000,1000,0).is_p1_win())
    # t1 > t2 < b1 > b2     player 1 wins 2000,0,1000,0
        self.assertTrue(get_t_wave(2000,0,1000,0).is_p1_win())

                            ## ><<  
    # t1 < t2 < b1 > b2     player 2 wins 0,500,2000,1000
        self.assertFalse(get_t_wave(0,500,2000,1000).is_p1_win())
    # t1 > t2 < b1 < b2     player 1 wins 2000,500,1000,1500
        self.assertTrue(get_t_wave(2000,500,1000,1500).is_p1_win())
    # t1 < t2 > b1 < b2     player 2 wins 0,1000,0,1000
        self.assertFalse(get_t_wave(0,1000,0,1000).is_p1_win())

                            ##   <=>
    # t1 < t2 = b1 > b2     player 2 wins 500,2000,2000,500
        self.assertFalse(get_t_wave(500,2000,2000,500).is_p1_win())
    # t1 < t2 = b1 > b2     player 2 wins 0,2000,2000,500
        self.assertFalse(get_t_wave(0,2000,2000,500).is_p1_win())
    # t1 < t2 = b1 > b2     player 1 wins 500,2000,2000,0
        self.assertTrue(get_t_wave(500,2000,2000,0).is_p1_win())

    # t1 < t2 > b1 = b2     player 2 wins 500,2000,500,500
        self.assertFalse(get_t_wave(500,2000,500,500).is_p1_win())
    # t1 = t2 > b1 < b2     player 2 wins 500,500,0,500 
        self.assertFalse(get_t_wave(500,500,0,500).is_p1_win())
    # t1 = t2 < b1 > b2     player 1 wins 500,500,2000,0
        self.assertTrue(get_t_wave(500,500,2000,0).is_p1_win())
    # t1 > t2 < b1 = b2     player 1 wins 2000,500,1000,1000
        self.assertTrue(get_t_wave(2000,500,1000,1000).is_p1_win())
    # t1 > t2 = b1 < b2     player 1 wins 2000,500,500,1000 (false)!
        self.assertTrue(get_t_wave(2000,500,500,1000).is_p1_win())
    # t1 > t2 = b1 < b2     player 1 wins 2000,0,0,1000
        self.assertTrue(get_t_wave(2000,0,0,1000).is_p1_win())


####################

    # t1 = t2 = b1 = b2     0,0,0,0
    # t1 = t2 = b1 = b2     2000,2000,2000,2000
    # t1 > t2 > b1 > b2     player 1 wins (true)
    # t1 < t2 < b1 < b2     player 2 wins (false)
                            ## >==
    # t1 > t2 = b1 = b2
    # t1 = t2 > b1 = b2
    # t1 = t2 = b1 > b2 
                            ## <==
    # t1 < t2 = b1 = b2 
    # t1 = t2 < b1 = b2 
    # t1 = t2 = b1 < b2
                            ## >>=
    # t1 > t2 > b1 = b2    
    # t1 = t2 > b1 > b2 
    # t1 > t2 = b1 > b2
                            ## <<=
    # t1 < t2 < b1 = b2
    # t1 < t2 = b1 < b2
    # t1 = t2 < b1 < b2
                            ## >><
    # t1 > t2 > b1 < b2 
    # t1 < t2 > b1 > b2 
    # t1 > t2 < b1 > b2
                            ## ><<  
    # t1 < t2 < b1 > b2 
    # t1 > t2 < b1 < b2
    # t1 < t2 > b1 < b2
                            ##   <=>
    # t1 < t2 = b1 > b2 
    # t1 < t2 > b1 = b2 
    # t1 = t2 > b1 < b2 
    # t1 = t2 < b1 > b2
    # t1 > t2 < b1 = b2 
    # t1 > t2 = b1 < b2 

    # def test_is_health_same(self):
    #     w = get_waves(1)
    #     # everybody loses all at same time, it's a tie
    #     w.top.nexus_self     = 0
    #     w.top.nexus_enemy    = 0
    #     w.bottom.nexus_self  = 0
    #     w.bottom.nexus_enemy = 0
    #     self.assertTrue(w.is_health_same())
        
    #     # everybody wins, its a tie
    #     w.top.nexus_self     = 2000
    #     w.top.nexus_enemy    = 2000
    #     w.bottom.nexus_self  = 2000
    #     w.bottom.nexus_enemy = 2000
    #     self.assertTrue(w.is_health_same())

    #     # true tie, both players did damage to eachother equally
    #     w.top.nexus_self     = 500
    #     w.top.nexus_enemy    = 500
    #     w.bottom.nexus_self  = 1800
    #     w.bottom.nexus_enemy = 1800
    #     self.assertTrue(w.is_health_same())

    #     # true tie, where each players nexus has different values
    #     w.top.nexus_self     = 500
    #     w.top.nexus_enemy    = 1800
    #     w.bottom.nexus_self  = 1800
    #     w.bottom.nexus_enemy = 500
    #     self.assertTrue(w.is_health_same()) #?

    #     # player 1  loses, bottom lane is destroyed
    #     w.top.nexus_self     = 2000
    #     w.top.nexus_enemy    = 2000
    #     w.bottom.nexus_self  = 0
    #     w.bottom.nexus_enemy = 2000
    #     self.assertFalse(w.is_health_same())
        
    #     # player 1 loses, top lane is destroyed
    #     w.top.nexus_self     = 0
    #     w.top.nexus_enemy    = 2000
    #     w.bottom.nexus_self  = 2000
    #     w.bottom.nexus_enemy = 2000
    #     self.assertFalse(w.is_health_same())

    #      # player 2  loses, bottom lane is destroyed
    #     w.top.nexus_self     = 2000
    #     w.top.nexus_enemy    = 2000
    #     w.bottom.nexus_self  = 2000
    #     w.bottom.nexus_enemy = 0
    #     self.assertFalse(w.is_health_same())
        
    #     # player 2 loses, top lane is destroyed
    #     w.top.nexus_self     = 2000
    #     w.top.nexus_enemy    = 0
    #     w.bottom.nexus_self  = 2000
    #     w.bottom.nexus_enemy = 2000
    #     self.assertFalse(w.is_health_same())

    #     # player 1 loses, bottom lane destroyed
    #     w.top.nexus_self     = 1800
    #     w.top.nexus_enemy    = 500
    #     w.bottom.nexus_self  = 0
    #     w.bottom.nexus_enemy = 500
    #     self.assertFalse(w.is_health_same())

    #     # player 1 loses, top lane destroyed
    #     w.top.nexus_self     = 0
    #     w.top.nexus_enemy    = 500
    #     w.bottom.nexus_self  = 1800
    #     w.bottom.nexus_enemy = 500

    #      # player 2 loses, bottom lane destroyed
    #     w.top.nexus_self     = 1800
    #     w.top.nexus_enemy    = 500
    #     w.bottom.nexus_self  = 500
    #     w.bottom.nexus_enemy = 0
    #     self.assertFalse(w.is_health_same())

    #     # player 2 loses, top lane destroyed
    #     w.top.nexus_self     = 500
    #     w.top.nexus_enemy    = 0
    #     w.bottom.nexus_self  = 1800
    #     w.bottom.nexus_enemy = 500


    #     self.assertFalse(w.is_health_same())




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
