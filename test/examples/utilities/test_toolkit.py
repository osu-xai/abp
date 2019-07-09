import sys
import unittest

import test_wave
import test_wave_group
import test_episode
import test_episodes

loader = unittest.TestLoader()
suite  = unittest.TestSuite()

suite.addTests(loader.loadTestsFromModule(test_wave))
suite.addTests(loader.loadTestsFromModule(test_wave_group))
suite.addTests(loader.loadTestsFromModule(test_episode))
suite.addTests(loader.loadTestsFromModule(test_episodes))


runner = unittest.TextTestRunner()
result = runner.run(suite)