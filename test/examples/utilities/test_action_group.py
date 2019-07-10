import sys
import unittest

sys.path.append('../../../abp/examples/pysc2/tug_of_war/utilities')

from abp.examples.pysc2.tug_of_war.utilities import action_group
from abp.examples.pysc2.tug_of_war.utilities import actions
from abp.examples.pysc2.tug_of_war.utilities import wave

import test_wave



empty_wave = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
empty_wave = wave.Wave(empty_wave)
waveGroup1 = test_wave.get_waves(3)
actionGroup1 = []
actionGroup1.append(actions.Actions(empty_wave,   waveGroup1[0]))
actionGroup1.append(actions.Actions(waveGroup1[0],waveGroup1[1]))
actionGroup1.append(actions.Actions(waveGroup1[1],waveGroup1[2]))



class TestWaveGroup(unittest.TestCase):

    def test_action_group(self):
        ag = action_group.ActionGroup(actionGroup1)
        
        self.assertEqual(ag.action_group_size, 3)
        
        for i in range(3):
            self.assertIsInstance(ag.actions[i], actions.Actions)

        move = ag.get_move_string()
        self.assertEqual(move, "Top: 101, 102, 103 Bottom: 104, 105, 106| Top: 100, 100, 100 Bottom: 100, 100, 100| Top: 100, 100, 100 Bottom: 100, 100, 100| ")

if __name__ == "__main__":
    unittest.main()