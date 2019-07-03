import sys
import unittest

sys.path.append('../../../abp/examples/pysc2/tug_of_war/')

from abp.examples.pysc2.tug_of_war import episode
from abp.examples.pysc2.tug_of_war import wave
import test_wave

episode1 = test_wave.get_waves(40)

class TestEpisode(unittest.TestCase):

    def test_episode(self):
        ep = episode.Episode(episode1, 5)
        
        self.assertEqual(ep.episode_size, 40)

        for i in range(40):
            self.assertIsInstance(ep.episode[i], wave.Wave)



if __name__ == "__main__":
    unittest.main()