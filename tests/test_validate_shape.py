import unittest
from tianshou_agents.components.env import is_valid_shape

class TestValidShape(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testShapeValid(self):
        self.assertTrue(is_valid_shape(4))
        self.assertTrue(is_valid_shape((1, 2, 3)))
        self.assertTrue(is_valid_shape([4, 1, 1]))
        self.assertTrue(is_valid_shape([(4, 4), 1, 1]))
        self.assertTrue(is_valid_shape(((4, 4), 1, 1)))

    def testShapeInvalid(self):
        self.assertFalse(is_valid_shape(None))
        self.assertFalse(is_valid_shape(1.5))
        self.assertFalse(is_valid_shape((1, 1.5, 1)))
        self.assertFalse(is_valid_shape(((4, [4, 4]), 1, 1)))
        self.assertFalse(is_valid_shape("ttt"))
        self.assertFalse(is_valid_shape((1, 2, "ttt")))
