import sys
import unittest

import test_wave
import test_action_group
import test_episode
import test_episodes
import test_actions

loader = unittest.TestLoader()
suite  = unittest.TestSuite()

suite.addTests(loader.loadTestsFromModule(test_wave))
suite.addTests(loader.loadTestsFromModule(test_action_group))
suite.addTests(loader.loadTestsFromModule(test_episode))
suite.addTests(loader.loadTestsFromModule(test_episodes))
suite.addTests(loader.loadTestsFromModule(test_actions))



runner = unittest.TextTestRunner()
result = runner.run(suite)