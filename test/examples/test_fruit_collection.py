import sys
import unittest

class FruitCollectionTests(unittest.TestCase):
    def test_fruit_collection_hra(self):
        sys.argv = ['',
                '--task', 'abp.examples.open_ai.fruit_collection.hra',
                '--folder', 'test/tasks/fruit_collection_hra_v2']
        from abp.trainer.task_runner import main
        main()

    def test_fruit_collection_dqn(self):
        sys.argv = ['',
                '--task', 'abp.examples.open_ai.fruit_collection.dqn',
                '--folder', 'test/tasks/fruit_collection_dqn_v1']
        from abp.trainer.task_runner import main
        main()
