import unittest
from tianshou_agents.components.env import is_valid_shape, extract_shape
import gym

class TestExtractShape(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self):
        pass        

    def test_extract_Discrete(self):
        space = gym.spaces.Discrete(10)
        self.assertEqual(extract_shape(space), 10)

    def test_extract_Box(self):
        space = gym.spaces.Box(low=0, high=1, shape=(10,))
        self.assertEqual(extract_shape(space), (10,))

    def test_extract_Tuple(self):
        space = gym.spaces.Tuple((gym.spaces.Discrete(10), gym.spaces.Box(low=0, high=1, shape=(10,))))
        self.assertEqual(extract_shape(space), (10, (10,)))

    def test_extract_Dict(self):
        space = gym.spaces.Dict({'a': gym.spaces.Discrete(10), 'b': gym.spaces.Box(low=0, high=1, shape=(10,))})
        self.assertEqual(extract_shape(space), {'a': 10, 'b': (10,)})

    def test_extract_MultiDiscrete(self):
        space = gym.spaces.MultiDiscrete([10, 10])
        self.assertEqual(extract_shape(space), (10, 10))

    def test_extract_MultiBinary(self):
        space = gym.spaces.MultiBinary(10)
        self.assertEqual(extract_shape(space), (10,))

    def test_extract_Other(self):
        class OtherSpace(gym.Space):
            pass

        space = OtherSpace()
        self.assertIs(extract_shape(space), space)

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

    def testShapeDepth(self):
        self.assertEqual(is_valid_shape(4, return_depth=True), (True, 0))
        self.assertEqual(is_valid_shape((4, 3, 2), return_depth=True), (True, 1))
        self.assertEqual(is_valid_shape((4, [3, 5], 2), return_depth=True), (True, 2))
        self.assertEqual(is_valid_shape((1.5, [3, 5], 2), return_depth=True), (False, 1))
