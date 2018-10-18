import sys
import unittest

class CartPoleTests(unittest.TestCase):
    def test_cart_pole_dqn(self):
        sys.argv = ['',
                '--task', 'abp.examples.open_ai.cart_pole.dqn',
                '--folder', 'test/tasks/cart_pole_dqn_v1']
        from abp.trainer.task_runner import main
        main()
