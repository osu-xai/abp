import sys
import unittest

sys.path.append('../../../abp/examples/pysc2/tug_of_war/utilities')

from abp.examples.pysc2.tug_of_war.utilities import episodes
from abp.examples.pysc2.tug_of_war.utilities import episode
from abp.examples.pysc2.tug_of_war.utilities import wave
from abp.examples.pysc2.tug_of_war.utilities import action_group

import test_wave


def create_episode_boundaries(raw_data):
    # divide data up into episodes of length 1,2,3,4... by zeroing out appropriate entries
    count = 0
    inserting = 0
    for d in range(len(raw_data)):
        if count == inserting:
            for i in range(len(raw_data[d][0])):
                raw_data[d][0][i] = 0
            inserting = count + 1
            count = 0
            continue
        count += 1
    return raw_data


class TestEpisodes(unittest.TestCase):
    def test_episodes(self):
        raw_data = test_wave.get_waves_raw_data(20)
        data_with_episodes = create_episode_boundaries(raw_data)
        eps = episodes.Episodes(data_with_episodes, 20, 1)

        self.assertEqual(len(eps.episodes), 4)
        self.assertIsInstance(eps.episodes[0], episode.Episode)

        self.assertEqual(eps.get_episode_count(), 4)
        
        self.assertIsInstance(eps.get_last_episode(), episode.Episode)
        self.assertEqual(eps.get_last_episode().episode_size, 4)

        for i in range(4):
            self.assertIsInstance(eps.get_episode(i), episode.Episode)
            self.assertEqual(eps.get_episode(i).episode_size, i+1)

    def test_get_total_p1_wins(self):
        raw_data = test_wave.get_waves_raw_data(20)
        data_with_episodes = create_episode_boundaries(raw_data)
        eps = episodes.Episodes(data_with_episodes, 20, 1)

        self.assertEqual(eps.get_total_p1_wins(), 0)
    

    def test_get_p1_buildings(self):
        raw_data = test_wave.get_waves_raw_data(6)
        data_with_episodes = create_episode_boundaries(raw_data)

        eps = episodes.Episodes(data_with_episodes, 6, 1)
        p1_buildings = eps.get_p1_buildings()

        self.assertEqual(p1_buildings[0], 1410) #losing mar
        self.assertEqual(p1_buildings[1], 1414) #losing bane
        self.assertEqual(p1_buildings[2], 1418) #losing imm
        self.assertEqual(p1_buildings[3], 0) #winning mar
        self.assertEqual(p1_buildings[4], 0) #winning bane
        self.assertEqual(p1_buildings[5], 0) #winning imm

    def test_get_p2_buildings(self):
        raw_data = test_wave.get_waves_raw_data(6)
        data_with_episodes = create_episode_boundaries(raw_data)

        eps = episodes.Episodes(data_with_episodes, 6, 1)
        p2_buildings = eps.get_p2_buildings()

        self.assertEqual(p2_buildings[0], 0) #losing mar
        self.assertEqual(p2_buildings[1], 0) #losing bane
        self.assertEqual(p2_buildings[2], 0) #losing imm
        self.assertEqual(p2_buildings[3], 1438) #winning mar
        self.assertEqual(p2_buildings[4], 1442) #winning bane
        self.assertEqual(p2_buildings[5], 1446) #winning imm
    
    def test_get_win_loss_sequence(self):
        raw_data = test_wave.get_waves_raw_data(20)
        data_with_episodes = create_episode_boundaries(raw_data)

        eps = episodes.Episodes(data_with_episodes, 20, 1)
        win_loss_sequence = eps.get_win_loss_sequence()
        
        self.assertEqual(len(win_loss_sequence), 5)
        for i in range(4):
            self.assertEqual(win_loss_sequence[i], (-1*i))

    def test_get_move_set(self):
        raw_data = test_wave.get_waves_raw_data(20)
        data_with_episodes = create_episode_boundaries(raw_data)

        eps = episodes.Episodes(data_with_episodes, len(data_with_episodes), 1)
    
        self.assertDictEqual(eps.get_binned_move_sets()[0], {'Top: 401, 402, 403 Bottom: 404, 405, 406| ': 1, 'Top: 201, 202, 203 Bottom: 204, 205, 206| ': 1, 'Top: 701, 702, 703 Bottom: 704, 705, 706| ': 1, 'Top: 1101, 1102, 1103 Bottom: 1104, 1105, 1106| ': 1})
        self.assertDictEqual(eps.get_binned_move_sets()[1], {'Top: 100, 100, 100 Bottom: 100, 100, 100| ': 3})
        self.assertDictEqual(eps.get_binned_move_sets()[2], {'Top: 100, 100, 100 Bottom: 100, 100, 100| ': 2})
        self.assertDictEqual(eps.get_binned_move_sets()[3], {'Top: 100, 100, 100 Bottom: 100, 100, 100| ': 1})

    def test_get_end_building_frequencies(self):
        raw_data = test_wave.get_waves_raw_data(5)
        data_with_episodes = create_episode_boundaries(raw_data)

        eps = episodes.Episodes(data_with_episodes, len(data_with_episodes), 1)
        marine_dict, baneling_dict, immortal_dict = eps.get_end_building_frequencies()
        self.assertDictEqual(marine_dict, {'Top Player 1 Marine (201)' : 1, 'Bottom Player 1 Marine (204)' : 1, 'Top Player 2 Marine (208)' : 1, 'Bottom Player 2 Marine (211)' : 1})
        self.assertDictEqual(baneling_dict, {'Top Player 1 Baneling (202)' : 1, 'Bottom Player 1 Baneling (205)' : 1, 'Top Player 2 Baneling (209)' : 1, 'Bottom Player 2 Baneling (212)' : 1})
        self.assertDictEqual(immortal_dict, {'Top Player 1 Immortal (203)' : 1, 'Bottom Player 1 Immortal (206)' : 1, 'Top Player 2 Immortal (210)' : 1, 'Bottom Player 2 Immortal (213)' : 1})


if __name__ == "__main__":
    unittest.main()

