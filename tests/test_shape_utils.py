import unittest
from tianshou.data import Batch
from tianshou_agents.components.env import (
    is_pure_shape, extract_shape, batch_flatten,
    construct_space, batch2tensor
)
import numpy as np
import torch
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
        self.assertEqual(extract_shape(space), 10)

    def test_extract_Tuple(self):
        space = gym.spaces.Tuple((gym.spaces.Discrete(10), gym.spaces.Box(low=0, high=1, shape=(10,))))
        self.assertEqual(extract_shape(space), (10, 10))

    def test_extract_Dict(self):
        space = gym.spaces.Dict({'a': gym.spaces.Discrete(10), 'b': gym.spaces.Box(low=0, high=1, shape=(10,))})
        self.assertEqual(extract_shape(space), {'a': 10, 'b': 10})

    def test_extract_MultiDiscrete(self):
        space = gym.spaces.MultiDiscrete([10, 10])
        self.assertEqual(extract_shape(space), (10, 10))

    def test_extract_MultiBinary(self):
        space = gym.spaces.MultiBinary(10)
        self.assertEqual(extract_shape(space), 10)

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
        self.assertTrue(is_pure_shape(4))
        self.assertTrue(is_pure_shape((1, 2, 3)))
        self.assertTrue(is_pure_shape([4, 1, 1]))
        self.assertTrue(is_pure_shape([(4, 4), 1, 1]))
        self.assertTrue(is_pure_shape(((4, 4), 1, 1)))

    def testShapeInvalid(self):
        self.assertFalse(is_pure_shape(None))
        self.assertFalse(is_pure_shape(1.5))
        self.assertFalse(is_pure_shape((1, 1.5, 1)))
        self.assertFalse(is_pure_shape(((4, [4, 4]), 1, 1)))
        self.assertFalse(is_pure_shape("ttt"))
        self.assertFalse(is_pure_shape((1, 2, "ttt")))

    def testShapeDepth(self):
        self.assertEqual(is_pure_shape(4, return_depth=True), (True, 0))
        self.assertEqual(is_pure_shape((4, 3, 2), return_depth=True), (True, 1))
        self.assertEqual(is_pure_shape((4, [3, 5], 2), return_depth=True), (True, 2))
        self.assertEqual(is_pure_shape((1.5, [3, 5], 2), return_depth=True), (False, 1))

class TestFlattenBatch(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testDiscrete(self):
        space = gym.spaces.Discrete(4)

        obs = np.array([
            space.sample() for i in range(5)
        ])

        self.assertEqual(batch_flatten(space, obs).shape, (5, 4))

    def testMultiDiscrete(self):
        space = gym.spaces.MultiDiscrete([3,4,5])

        obs = np.array([
            space.sample() for i in range(5)
        ])
        
        self.assertEqual(batch_flatten(space, obs).shape, (5, 12))

    def testDictSpace(self):
        space = gym.spaces.Dict({
            'part1': gym.spaces.Box(
                    low=-1, high=1, shape=(2,), dtype=np.float32
            ),
            'part2': gym.spaces.Box(
                low=-8, high=8, shape=(1,), dtype=np.float32
            )    
        })

        obs = Batch(
            part1 = np.array([
                [
                    space['part1'].sample()
                        for i in range(5)
                ]
            ]),
            part2 = np.array([
                [
                    space['part2'].sample()
                        for i in range(5)
                ]
            ]),
        )

        self.assertEqual(batch_flatten(space, obs).shape, (5, 3))

    def testTupleSpace(self):
        space = gym.spaces.Tuple([
            gym.spaces.Box(
                low=-1, high=1, shape=(2,), dtype=np.float32
            ), gym.spaces.Box(
                low=-8, high=8, shape=(1,), dtype=np.float32
            )
        ])

        obs = np.array([
            [ 
                space.spaces[0].sample(),
                space.spaces[1].sample()
            ] for i in range(5)
        ], dtype=object)

        self.assertEqual(batch_flatten(space, obs).shape, (5, 3))

    def testBoxSpace(self):
        space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        obs = np.asarray([
            space.sample() for i in range(5)
        ])

        self.assertEqual(batch_flatten(space, obs).shape, (5, 2))

    def testBox2Space(self):
        space = gym.spaces.Box(low=-1, high=1, shape=(3,2), dtype=np.float32)

        obs = np.asarray([
            space.sample() for i in range(5)
        ])

        self.assertEqual(batch_flatten(space, obs).shape, (5, 6))

class TestConstructSpace(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testIntTuple(self):
        shape = (5, 4, 3)
        space = construct_space(shape)        
        self.assertEqual(
            space,
            gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=shape, dtype=np.float32
            )
        )

    def testNestedTuple(self):
        shape = ((5, 4, 3), (4, 2))
        space = construct_space(shape)
        self.assertEqual(
            space,
            gym.spaces.Tuple(
                [
                    gym.spaces.Box(
                        low=-np.inf, high=np.inf,
                        shape=shape[0], dtype=np.float32
                    ),
                    gym.spaces.Box(
                        low=-np.inf, high=np.inf,
                        shape=shape[1], dtype=np.float32
                    ),
                ]
            )
        )

    def testNestedDict(self):
        shape = {'part1': (5, 4, 3), 'part2': (4, 2)}
        space = construct_space(shape)
        self.assertEqual(
            space,
            gym.spaces.Dict({
                'part1': gym.spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=shape['part1'], dtype=np.float32
                ),
                'part2': gym.spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=shape['part2'], dtype=np.float32)
                }
            )
        )

    def testInt(self):
        shape = 5
        space = construct_space(shape)
        self.assertEqual(
            space,
            gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(shape,), dtype=np.float32
            )
        )

class TestBatch2Tensor(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testDiscrete(self):
        space = gym.spaces.Discrete(4)
        obs = np.array([0, 3, 3, 0, 0])
        t = batch2tensor(space, obs)
        self.assertTrue(torch.equal(t, torch.as_tensor([
            [1., 0., 0., 0.],
            [0., 0., 0., 1.],
            [0., 0., 0., 1.],
            [1., 0., 0., 0.],
            [1., 0., 0., 0.]
        ])))

    def testMultiDiscrete(self):
        space = gym.spaces.MultiDiscrete([2,3,3])

        obs = np.array([
            [1, 2, 2],
            [1, 2, 0],
            [1, 0, 2]
        ], dtype=np.int64)

        t = batch2tensor(space, obs)
        self.assertEqual(len(t), 3)

        self.assertTrue(torch.equal(t[0],
            torch.tensor([
                [0., 1.],
                [0., 1.],
                [0., 1.]
            ])
        ))

        self.assertTrue(torch.equal(t[1],
            torch.tensor([
                [0., 0., 1.],
                [0., 0., 1.],
                [1., 0., 0.]
            ])
        ))

        self.assertTrue(torch.equal(t[2],
            torch.tensor([
                [0., 0., 1.],
                [1., 0., 0.],
                [0., 0., 1.]
            ])
        ))

    def testDictSpace(self):
        space = gym.spaces.Dict({
            'part1': gym.spaces.Box(
                    low=-1, high=1, shape=(2,), dtype=np.float32
            ),
            'part2': gym.spaces.Box(
                low=-8, high=8, shape=(1,), dtype=np.float32
            )    
        })

        obs = Batch({
            'part1': np.array([[
                [-0.5604604 , -0.12710193],
                [-0.9924539 ,  0.5162854 ],
                [-0.2942568 , -0.6883518 ],
                [-0.90841126,  0.7958541 ],
                [-0.29710975,  0.7938635 ]
            ]], dtype=np.float32),
            'part2': np.array([[
                [-0.63503754],
                [-1.162916  ],
                [-1.2045231 ],
                [-0.09973483],
                [ 5.1034975 ]
            ]], dtype=np.float32),
        })

        t = batch2tensor(space, obs)

        self.assertEqual(len(t.keys()), 2)

        self.assertTrue(torch.equal(
            t['part1'],
            torch.tensor([
                [-0.5604604 , -0.12710193],
                [-0.9924539 ,  0.5162854 ],
                [-0.2942568 , -0.6883518 ],
                [-0.90841126,  0.7958541 ],
                [-0.29710975,  0.7938635 ]
            ])
        ))

        self.assertTrue(torch.equal(
            t['part2'],
            torch.tensor([
                [-0.63503754],
                [-1.162916  ],
                [-1.2045231 ],
                [-0.09973483],
                [ 5.1034975 ]
            ])
        ))

    def testTupleSpace(self):
        space = gym.spaces.Tuple([
            gym.spaces.Box(
                low=-1, high=1, shape=(2,), dtype=np.float32
            ), gym.spaces.Box(
                low=-8, high=8, shape=(1,), dtype=np.float32
            )
        ])

        obs = np.array([[
            np.array([ 0.0651449, -0.3374453], dtype=np.float32),
            np.array([-0.58095586], dtype=np.float32)
        ], [
            np.array([-0.20044582,  0.1049023 ], dtype=np.float32),
            np.array([6.8833327], dtype=np.float32)
        ], [
            np.array([0.88469183, 0.09370121], dtype=np.float32),
            np.array([3.8046243], dtype=np.float32)
        ]], dtype=object)

        t = batch2tensor(space, obs)

        self.assertEqual(len(t), 2)

        self.assertTrue(torch.equal(
            t[0],
            torch.tensor([
                [0.0651449, -0.3374453],
                [-0.20044582,  0.1049023 ],
                [0.88469183, 0.09370121]
            ])
        ))

        self.assertTrue(torch.equal(
            t[1],
            torch.tensor([
                [-0.58095586],
                [6.8833327],
                [3.8046243]
            ])
        ))

    def testBoxSpace(self):
        space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        obs = np.array([
            [-0.28892794, -0.23318008],
            [-0.80961716,  0.4208645 ],
            [ 0.12712371, -0.53597057],
            [-0.60322696,  0.6333279 ],
            [-0.53961515, -0.647485  ]
        ], dtype=np.float32)

        t = batch2tensor(space, obs)

        self.assertTrue(torch.equal(
            t,
            torch.tensor([
                [-0.28892794, -0.23318008],
                [-0.80961716,  0.4208645 ],
                [ 0.12712371, -0.53597057],
                [-0.60322696,  0.6333279 ],
                [-0.53961515, -0.647485  ]
            ])
        ))