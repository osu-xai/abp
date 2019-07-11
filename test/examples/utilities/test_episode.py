import sys
import unittest

sys.path.append('../../../abp/examples/pysc2/tug_of_war/')

from abp.examples.pysc2.tug_of_war.utilities import episode
from abp.examples.pysc2.tug_of_war.utilities import wave
from abp.examples.pysc2.tug_of_war.utilities import action_group
from abp.examples.pysc2.tug_of_war.utilities import actions


import test_wave


class TestEpisode(unittest.TestCase):
    def test_episode(self):
        ep = episode.Episode(test_wave.get_waves(40), 5)
        self.assertEqual(ep.episode_size, 40)
        for i in range(40):
            self.assertIsInstance(ep.episode_waves[i], wave.Wave)

    # check for correct containerization
    def test_get_action_groups(self):
        ep = episode.Episode(test_wave.get_waves(40), 5)
        for wg in ep.get_action_groups():
            self.assertIsInstance(wg, action_group.ActionGroup)
            self.assertEqual(wg.action_group_size, 5)

    # episode is shorter than group size
    def test_get_action_groups_A(self):
        try:
            ep = episode.Episode(test_wave.get_waves(4), 5)
            self.assertFail(msg="should have thrown exception for groupsize > episode length ")
        except:
            self.assertTrue(True)

    # episode is same as group size
    def test_get_action_groups_B(self):
        try:
            ep = episode.Episode(test_wave.get_waves(4), 4)
            self.assertEqual(ep.get_action_group_count(), 1)
        except:
            self.assertFail(msg="episode same as group size should not have thrown exception")

    # episode is between one and two group sizes
    def test_get_action_groups_C(self):
        try:
            ep = episode.Episode(test_wave.get_waves(14), 8)
            self.assertEqual(ep.get_action_group_count(), 1)
        except:
            self.assertFail(msg="Should not have thrown exception for episode is between one two group sizes")

    # episode is multiple groups evenly
    def test_get_action_groups_D(self):
        try:
            ep = episode.Episode(test_wave.get_waves(40), 5)
            self.assertEqual(ep.get_action_group_count(), 8)
        except:
            self.assertFail(msg="Should not have thrown exception for episode is multiple groups evenly")

    # episode is one wave more than even count
    def test_get_action_groups_E(self):
        try:
            ep = episode.Episode(test_wave.get_waves(41), 5)
            self.assertEqual(ep.get_action_group_count(), 8)
        except:
            self.assertFail(msg="Should not have thrown exception for episode is one wave more than even count")

    # group size is 0
    def test_create_episode_group_length_0(self):
        try:
            ep = episode.Episode(test_wave.get_waves(4), 0)
            self.assertFail(msg="should have thrown exception for group length = 0 ")
        except:
            self.assertTrue(True)


    # group size is 1
    def test_create_episode_group_length_1(self):
        try:
            ep = episode.Episode(test_wave.get_waves(17), 1)
            self.assertEqual(ep.get_action_group_count(), 17)
        except:
            self.assertFail(msg="Should not have thrown exception for episode is one wave more than even count")

    # group size is 2
    def test_create_episode_group_length_2(self):
        try:
            ep = episode.Episode(test_wave.get_waves(17), 2)
            self.assertEqual(ep.get_action_group_count(), 8)
        except:
            self.assertFail(msg="Should not have thrown exception for episode is one wave more than even count")

    # episode size is 0
    def test_create_episode_length_0(self):
        try:
            ep = episode.Episode(test_wave.get_waves(0), 4)
            self.assertFail(msg="should have thrown exception for episode size = 0 ")
        except:
            self.assertTrue(True)

    # group size is -1, episode is x
    def test_create_episode_group_length_neg(self):
        try:
            ep = episode.Episode(test_wave.get_waves(20), -1)
            self.assertFail(msg="should have thrown exception for group length < 0 ")
        except:
            self.assertTrue(True)
            


    ## changing both group size and episode length knobs


    # group size is 4, episode size is 4
    def test_get_action_groups_f(self):
        try:
            ep = episode.Episode(test_wave.get_waves(4), 4)
            self.assertEqual(ep.get_action_group_count(), 1)
        except:
            self.assertFail(msg="Should not have thrown exception for episode is one wave more than even count")

    # group size is 4, episode size is 3
    def test_create_episode_group_length_gt_episode_length(self):
        try:
            ep = episode.Episode(test_wave.get_waves(3), 4)
            self.assertFail(msg="should have thrown exception for group length > episode length ")
        except:
            self.assertTrue(True)
            
    # group size is 4, episode size is 5
    def test_create_episode_group_length_lt_episode_length(self):
        try:
            ep = episode.Episode(test_wave.get_waves(5), 4)
            self.assertEqual(ep.get_action_group_count(), 1)
        except:
            self.assertFail(msg="Should not have thrown exception for episode is one wave more than even count")


    # group size is 10, episode size is 101
    def test_create_episode_group_length_and_episode_length_large(self):
        try:
            ep = episode.Episode(test_wave.get_waves(101), 10)
            self.assertEqual(ep.get_action_group_count(), 10)
        except:
            self.assertFail(msg="Should not have thrown exception for episode is one wave more than even count")


    def test_get_last_episode_wave(self):
        ep = episode.Episode(test_wave.get_waves(40), 5)

        self.assertIsInstance(ep.get_last_episode_wave(), wave.Wave)

    
    def test_get_start_episode_wave(self):
        ep = episode.Episode(test_wave.get_waves(40), 5)

        self.assertIsInstance(ep.get_start_episode_wave(), wave.Wave)
    
    def test_create_move_dictionary(self):
        print()


if __name__ == "__main__":
    unittest.main()