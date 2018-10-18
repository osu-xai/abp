import sys
import unittest

class ScaiiRTSTests(unittest.TestCase):
    def test_multi_step_hra(self):
        sys.argv = ['',
                '--task', 'abp.examples.scaii.multi_step.hra',
                '--folder', 'test/tasks/sky_rts_multi_step_hra']
        from abp.trainer.task_runner import main
        main()
