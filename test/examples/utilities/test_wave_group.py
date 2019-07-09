import sys
import unittest

sys.path.append('../../../abp/examples/pysc2/tug_of_war/utilities')

from abp.examples.pysc2.tug_of_war.utilities import wave_group
from abp.examples.pysc2.tug_of_war.utilities import wave
import test_wave




waveGroup1 = test_wave.get_waves(5)


class TestWaveGroup(unittest.TestCase):

    def test_wave_group(self):
        wg = wave_group.WaveGroup(waveGroup1)
        
        self.assertEqual(wg.wave_group_size, 5)
        
        for i in range(5):
            self.assertIsInstance(wg.waves[i], wave.Wave)

            

if __name__ == "__main__":
    unittest.main()