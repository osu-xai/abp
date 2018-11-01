import sys
import unittest

class PySC2Tests(unittest.TestCase):
    def test_pysc2_shards_dqn(self):
        sys.argv = ['',
                '--task', 'abp.examples.pysc2.collect_shards.dqn',
                '--folder', 'test/tasks/pysc2_collect_shards_dqn']
        from abp.trainer.task_runner import main
        main()

    def test_pysc2_shards_hra(self):
        sys.argv = ['',
                '--task', 'abp.examples.pysc2.collect_shards.hra',
                '--folder', 'test/tasks/pysc2_collect_shards_hra']
        from abp.trainer.task_runner import main
        main()
