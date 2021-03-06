import sys
import unittest
import numpy as np


sys.path.append('../../examples/pysc2/tug_of_war/')
from abp.examples.pysc2.tug_of_war import gatherdata

a_mar = 0
a_vik = 1
a_col = 2
a_pyl = 3
a_nex = 4
e_mar = 5
e_vik = 6
e_col = 7
e_pyl = 8
e_nex = 9
 
data1 = np.array(
    [
        [
            [0,0,0,0,2000,0,0,0,0,2000,0,0,0,0,0,0,0],
            [1,0,0,0,2000,2,0,0,0,2000,1,0,0,0,0,0,0]
        ],
        [
            [1,0,0,0,2000,2,0,0,0,2000,1,0,0,0,0,0,0],
            [1,1,0,0,1950,3,0,0,0,2000,0,0,0,0,0,0,0]
        ],
        [
            [1,1,0,0,1500,3,0,0,0,2000,0,0,0,0,0,0,0],
            [1,6,0,0,1000,3,0,2,1,2000,0,0,0,0,0,0,0]
        ],
        [
            [1,6,0,0,1000,3,0,2,1,2000,0,0,0,0,0,0,0],
            [1,8,0,0,500,3,0,2,3,2000,0,0,0,0,0,0,0]
        ],
        [
            [1,8,0,0,500,3,0,2,3,2000,0,0,0,0,0,0,0],
            [4,8,0,0,100,4,0,2,3,2000,0,0,0,0,0,0,0]
        ],
        [
            [0,0,0,0,2000,0,0,0,0,2000,0,0,0,0,0,0,0],
            [1,0,0,0,2000,2,0,0,0,2000,1,0,0,0,0,0,0]
        ]
    ]
)


data2 = np.array(
    [
        [
            [0,0,0,0,2000,0,0,0,0,2000,0,0,0,0,0,0,0],
            [1,0,0,0,2000,2,0,0,0,2000,1,0,0,0,0,0,0]
        ],
        [
            [1,0,0,0,2000,2,0,0,0,2000,1,0,0,0,0,0,0],
            [1,1,0,0,1950,3,0,0,0,2000,0,0,0,0,0,0,0]
        ],
        [
            [1,1,0,0,1500,3,0,0,0,2000,0,0,0,0,0,0,0],
            [1,6,0,0,1000,3,0,2,1,2000,0,0,0,0,0,0,0]
        ],
        [
            [1,6,0,0,1000,3,0,2,1,2000,0,0,0,0,0,0,0],
            [1,8,0,0,500,3,0,2,3,2000,0,0,0,0,0,0,0]
        ],
        [
            [1,8,0,0,500,3,0,2,3,2000,0,0,0,0,0,0,0],
            [4,8,0,0,100,4,0,2,3,2000,0,0,0,0,0,0,0]
        ],
        #################### episode ###################
        [
            [0,0,0,0,2000,0,0,0,0,2000,0,0,0,0,0,0,0],
            [1,0,0,0,2000,2,0,0,0,2000,1,0,0,0,0,0,0]
        ],
        [
            [0,0,0,0,2000,0,0,0,0,2000,0,0,0,0,0,0,0],
            [1,0,0,0,2000,2,0,0,0,2000,1,0,0,0,0,0,0]
        ],
        [
            [1,0,0,0,2000,0,0,0,0,1900,1,0,0,0,0,0,0],
            [1,1,0,0,1950,1,0,0,0,2000,0,0,0,0,0,0,0]
        ],
        [
            [1,1,0,0,1500,1,0,0,0,1800,0,0,0,0,0,0,0],
            [1,6,0,0,1000,1,0,2,1,2000,0,0,0,0,0,0,0]
        ],
        [
            [1,6,0,0,1000,1,0,2,1,1500,0,0,0,0,0,0,0],
            [1,8,0,0,500,3,0,2,3,2000,0,0,0,0,0,0,0]
        ],
        [
            [9,7,4,1,1900,3,0,2,3,100,0,0,0,0,0,0,0],
            [4,8,0,0,100,4,0,2,3,9,0,0,0,0,0,0,0]
        ],
        #################### episode ####################
        [
            [0,0,0,0,2000,0,0,0,0,2000,0,0,0,0,0,0,0],
            [3,0,0,0,2000,4,0,0,0,2000,1,0,0,0,0,0,0]
        ],
        [
            [3,0,0,0,2000,4,0,0,0,2000,1,0,0,0,0,0,0],
            [4,1,0,0,1950,7,0,0,0,2000,0,0,0,0,0,0,0]
        ],
        [
            [4,1,0,0,1950,7,0,0,0,2000,0,0,0,0,0,0,0],
            [5,6,0,0,1000,10,0,2,1,2000,0,0,0,0,0,0,0]
        ],
        [
            [5,6,0,0,1000,10,0,2,1,2000,0,0,0,0,0,0,0],
            [5,8,0,0,500,10,0,2,3,2000,0,0,0,0,0,0,0]
        ],
        [
            [5,8,0,0,500,10,0,2,3,2000,0,0,0,0,0,0,0],
            [8,8,0,0,100,11,0,2,3,2000,0,0,0,0,0,0,0]
        ],
    ]
)


data3 = np.array(
    [
        [
            [0,0,0,0,2000,0,0,0,0,2000,0,0,0,0,0,0,0],
            [1,0,0,0,2000,2,0,0,0,2000,1,0,0,0,0,0,0]
        ],
        [
            [1,0,0,0,2000,2,0,0,0,2000,1,0,0,0,0,0,0],
            [1,1,0,0,1950,3,0,0,0,2000,0,0,0,0,0,0,0]
        ],
        [
            [1,1,0,0,1500,3,0,0,0,2000,0,0,0,0,0,0,0],
            [1,6,0,0,1000,3,0,2,1,2000,0,0,0,0,0,0,0]
        ],
        [
            [1,6,0,0,1000,3,0,2,1,2000,0,0,0,0,0,0,0],
            [1,8,0,0,500,3,0,2,3,2000,0,0,0,0,0,0,0]
        ],
        [
            [1,8,0,0,500,3,0,2,3,2000,0,0,0,0,0,0,0],
            [4,8,0,0,100,4,0,2,3,2000,0,0,0,0,0,0,0]
        ],
        #################### episode ###################
        [
            [0,0,0,0,2000,0,0,0,0,2000,0,0,0,0,0,0,0],
            [1,0,0,0,2000,2,0,0,0,2000,1,0,0,0,0,0,0]
        ],
        [
            [0,0,0,0,2000,0,0,0,0,2000,0,0,0,0,0,0,0],
            [1,0,0,0,2000,2,0,0,0,2000,1,0,0,0,0,0,0]
        ],
        [
            [1,0,0,0,2000,0,0,0,0,1900,1,0,0,0,0,0,0],
            [1,1,0,0,1950,1,0,0,0,2000,0,0,0,0,0,0,0]
        ],
        [
            [1,1,0,0,1500,1,0,0,0,1800,0,0,0,0,0,0,0],
            [1,6,0,0,1000,1,0,2,1,2000,0,0,0,0,0,0,0]
        ],
        [
            [1,6,0,0,1000,1,0,2,1,1500,0,0,0,0,0,0,0],
            [1,8,0,0,500,3,0,2,3,2000,0,0,0,0,0,0,0]
        ],
        [
            [9,7,4,1,1900,3,0,2,3,100,0,0,0,0,0,0,0],
            [4,8,0,0,1900,4,0,2,3,10,0,0,0,0,0,0,0]
        ],
        [
            [0,0,0,0,2000,0,0,0,0,2000,0,0,0,0,0,0,0],
            [1,0,0,0,2000,2,0,0,0,2000,1,0,0,0,0,0,0]
        ],
        #################### episode ####################
        [
            [0,0,0,0,2000,0,0,0,0,2000,0,0,0,0,0,0,0],
            [4,0,0,0,2000,3,0,0,0,2000,1,0,0,0,0,0,0]
        ],
        [
            [4,0,0,0,2000,3,0,0,0,2000,1,0,0,0,0,0,0],
            [7,0,0,0,2000,4,1,0,0,1950,0,0,0,0,0,0,0]
        ],
        [
            [7,0,0,0,2000,4,1,0,0,1950,0,0,0,0,0,0,0],
            [10,0,2,1,2000,5,6,0,0,1000,0,0,0,0,0,0,0]
        ],
        [
            [10,0,2,1,2000,5,6,0,0,1000,0,0,0,0,0,0,0],
            [10,0,2,3,2000,5,8,0,0,500,0,0,0,0,0,0,0]
        ],
        [
            [10,0,2,3,2000,5,8,0,0,500,0,0,0,0,0,0,0],
            [11,0,2,3,2000,8,8,0,0,100,0,0,0,0,0,0,0]
        ],
        [
            [0,0,0,0,2000,0,0,0,0,2000,0,0,0,0,0,0,0],
            [1,0,0,0,2000,2,0,0,0,2000,1,0,0,0,0,0,0]
        ]
    ]
)


data4 = np.array(
    [
        [
            [0,0,0,0,2000,0,0,0,0,2000,0,0,0,0,0,0,0],
            [1,0,0,0,2000,2,0,0,0,2000,1,0,0,0,0,0,0]
        ],
        [
            [1,0,0,0,2000,2,0,0,0,2000,1,0,0,0,0,0,0],
            [1,1,0,0,1950,3,0,0,0,2000,0,0,0,0,0,0,0]
        ],
        [
            [1,1,0,0,1500,3,0,0,0,2000,0,0,0,0,0,0,0],
            [1,6,0,0,1000,3,0,2,1,2000,0,0,0,0,0,0,0]
        ],
        [
            [1,6,0,0,1000,3,0,2,1,2000,0,0,0,0,0,0,0],
            [1,8,0,0,500,3,0,2,3,2000,0,0,0,0,0,0,0]
        ],
        [
            [1,8,0,0,500,3,0,2,3,2000,0,0,0,0,0,0,0],
            [4,8,0,0,100,4,0,2,3,2000,0,0,0,0,0,0,0]
        ],
        #################### episode ###################
        [
            [0,0,0,0,2000,0,0,0,0,2000,0,0,0,0,0,0,0],
            [1,0,0,0,2000,2,0,0,0,2000,1,0,0,0,0,0,0]
        ],
        [
            [1,0,0,0,2000,2,0,0,0,2000,1,0,0,0,0,0,0],
            [1,1,0,0,1950,3,0,0,0,2000,0,0,0,0,0,0,0]
        ],
        [
            [1,1,0,0,1500,3,0,0,0,2000,0,0,0,0,0,0,0],
            [1,6,0,0,1000,3,0,2,1,2000,0,0,0,0,0,0,0]
        ],
        [
            [1,6,0,0,1000,3,0,2,1,2000,0,0,0,0,0,0,0],
            [1,8,0,0,500,3,0,2,3,2000,0,0,0,0,0,0,0]
        ],
        [
            [1,8,0,0,500,3,0,2,3,2000,0,0,0,0,0,0,0],
            [4,8,0,0,100,4,0,2,3,2000,0,0,0,0,0,0,0]
        ],
         [
            [0,0,0,0,2000,0,0,0,0,2000,0,0,0,0,0,0,0],
            [1,0,0,0,2000,2,0,0,0,2000,1,0,0,0,0,0,0]
        ],
        [
            [1,0,0,0,2000,2,0,0,0,2000,1,0,0,0,0,0,0],
            [1,1,0,0,1950,3,0,0,0,2000,0,0,0,0,0,0,0]
        ],
        [
            [1,1,0,0,1500,3,0,0,0,2000,0,0,0,0,0,0,0],
            [1,6,0,0,1000,3,0,2,1,2000,0,0,0,0,0,0,0]
        ],
        [
            [1,6,0,0,1000,3,0,2,1,2000,0,0,0,0,0,0,0],
            [1,8,0,0,500,3,0,2,3,2000,0,0,0,0,0,0,0]
        ],
        [
            [1,8,0,0,500,3,0,2,3,2000,0,0,0,0,0,0,0],
            [4,8,0,0,100,4,0,2,3,2000,0,0,0,0,0,0,0]
        ]
    ]
)
class TestGatherData(unittest.TestCase):

    def test_detect_episode(self):
        ally_wins, enemy_wins, sum_ally_units_win, sum_enemy_units_win, win_total_timeline, sorted_refined_strats, sorted_episodes_length = gatherdata.gather_data(data1, 0, 0, 0, 0, 0, len(data1),[])
        
        self.assertEqual(ally_wins, 0)
        self.assertEqual(enemy_wins, 1)

    def test_get_unit_sums_of_winner(self):
        ally_wins, enemy_wins, sum_ally_units_win, sum_enemy_units_win, win_total_timeline, sorted_refined_strats, sorted_episodes_length = gatherdata.gather_data(data1, 0, 0, 0, 0, 0, len(data1),[])

        self.assertEqual(sum_enemy_units_win[a_mar]/enemy_wins, 4)
        self.assertEqual(sum_enemy_units_win[a_vik]/enemy_wins, 8)
        self.assertEqual(sum_enemy_units_win[a_col]/enemy_wins, 0)
        self.assertEqual(sum_enemy_units_win[a_pyl]/enemy_wins, 0)
        self.assertEqual(sum_enemy_units_win[a_nex]/enemy_wins, 100)
        self.assertEqual(sum_enemy_units_win[e_mar]/enemy_wins, 4)
        self.assertEqual(sum_enemy_units_win[e_vik]/enemy_wins, 0)
        self.assertEqual(sum_enemy_units_win[e_col]/enemy_wins, 2)
        self.assertEqual(sum_enemy_units_win[e_pyl]/enemy_wins, 3)
        self.assertEqual(sum_enemy_units_win[e_nex]/enemy_wins, 2000)
        

    def test_get_average_units_of_enemy_winner(self):
        ally_wins, enemy_wins, sum_ally_units_win, sum_enemy_units_win, win_total_timeline, sorted_refined_strats, sorted_episodes_length = gatherdata.gather_data(data2, 0, 0, 0, 0, 0, len(data2),[])
        
        self.assertEqual(sum_enemy_units_win[a_mar]/enemy_wins, 6)
        self.assertEqual(sum_enemy_units_win[a_vik]/enemy_wins, 8)
        self.assertEqual(sum_enemy_units_win[a_col]/enemy_wins, 0)
        self.assertEqual(sum_enemy_units_win[a_pyl]/enemy_wins, 0)
        self.assertEqual(sum_enemy_units_win[a_nex]/enemy_wins, 100)
        self.assertEqual(sum_enemy_units_win[e_mar]/enemy_wins, 7.5)
        self.assertEqual(sum_enemy_units_win[e_vik]/enemy_wins, 0)
        self.assertEqual(sum_enemy_units_win[e_col]/enemy_wins, 2)
        self.assertEqual(sum_enemy_units_win[e_pyl]/enemy_wins, 3)
        self.assertEqual(sum_enemy_units_win[e_nex]/enemy_wins, 2000)
    
    def test_get_average_units_of_enemy_winner(self):
        ally_wins, enemy_wins, sum_ally_units_win, sum_enemy_units_win, win_total_timeline, sorted_refined_strats, sorted_episodes_length = gatherdata.gather_data(data3, 0, 0, 0, 0, 0, len(data3),[])
        
        self.assertEqual(sum_ally_units_win[a_mar]/ally_wins, 7.5)
        self.assertEqual(sum_ally_units_win[a_vik]/ally_wins, 4)
        self.assertEqual(sum_ally_units_win[a_col]/ally_wins, 1)
        self.assertEqual(sum_ally_units_win[a_pyl]/ally_wins, 1.5)
        self.assertEqual(sum_ally_units_win[a_nex]/ally_wins, 1950)
        self.assertEqual(sum_ally_units_win[e_mar]/ally_wins, 6)
        self.assertEqual(sum_ally_units_win[e_vik]/ally_wins, 4)
        self.assertEqual(sum_ally_units_win[e_col]/ally_wins, 1)
        self.assertEqual(sum_ally_units_win[e_pyl]/ally_wins, 1.5)
        self.assertEqual(sum_ally_units_win[e_nex]/ally_wins, 55)
    
    def test_seperate_strategy(self):
        ally_wins, enemy_wins, sum_ally_units_win, sum_enemy_units_win, win_total_timeline, sorted_refined_strats, sorted_episodes_length  = gatherdata.gather_data(data4, 0, 0, 0, 0, 0, len(data4),[])
        print(sorted_refined_strats)
        self


if __name__ == "__main__":
    unittest.main()






    
