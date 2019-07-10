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
            for i in range(len(raw_data[d])):
                raw_data[d][i] = 0
            inserting = count + 1
            count = 0
            continue
        count += 1
    # print(raw_data)
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

        eps = episodes.Episodes(data_with_episodes, 20, 1)


if __name__ == "__main__":
    unittest.main()

