import sys
import unittest

sys.path.append('../../../abp/examples/pysc2/tug_of_war/')

from abp.examples.pysc2.tug_of_war.utilities import episode
from abp.examples.pysc2.tug_of_war.utilities import wave
from abp.examples.pysc2.tug_of_war.utilities import actions

import test_wave


class TestActions(unittest.TestCase):

    def test_get_actions(self):
        w = test_wave.get_waves(2)
        prev_wave = w[0]
        curr_wave = w[1]

        a = actions.Actions(prev_wave, curr_wave)
        self.assertEqual(a.top_self_marine, 100)
        self.assertEqual(a.top_self_baneling, 100)
        self.assertEqual(a.top_self_immortal, 100)
        self.assertEqual(a.bottom_self_marine, 100)
        self.assertEqual(a.bottom_self_baneling, 100)
        self.assertEqual(a.bottom_self_immortal, 100)

        self.assertEqual(a.top_enemy_marine, 100)
        self.assertEqual(a.top_enemy_baneling, 100)
        self.assertEqual(a.top_enemy_immortal, 100)
        self.assertEqual(a.bottom_enemy_marine, 100)
        self.assertEqual(a.bottom_enemy_baneling, 100)
        self.assertEqual(a.bottom_enemy_immortal, 100)
    


if __name__ == "__main__":
    unittest.main()
