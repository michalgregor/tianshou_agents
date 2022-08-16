import unittest

from tianshou.env.venvs import DummyVectorEnv
from tianshou_agents.methods.dqn import dqn_classic
from tianshou_agents.methods.sac import sac_classic
import numpy as np
import gym

# suppress tqdm output
import os
from functools import partialmethod
from tqdm import tqdm
tqdm.__init__ = partialmethod(tqdm.__init__, file=open(os.devnull, 'w'))

class PendulumDict(gym.Wrapper):
    def __init__(self):
        env = gym.make('Pendulum-v1')
        super().__init__(env)

        self.observation_space = gym.spaces.Dict({
            'part1': gym.spaces.Box(
                    low=-1, high=1, shape=(2,), dtype=np.float32
            ),
            'part2': gym.spaces.Box(
                low=-8, high=8, shape=(1,), dtype=np.float32
            )    
        })

    def split_obs(self, obs):
        return {'part1': obs[:2], 'part2': obs[-1:]}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.split_obs(obs), reward, done, info

    def reset(self):
        return self.split_obs(self.env.reset())

gym.register(
    id='PendulumDict-v0',
    entry_point="test_agents:PendulumDict",
    max_episode_steps=200
)

class TestAgentMixin:
    TaskName = ''
    AgentPreset = None
    Task = None

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testAgentTrainAndTest(self):
        agent = self.AgentPreset(
            self.TaskName,
            task=self.Task, max_epoch=2,
            step_per_epoch=5,
            train_envs=5,
            test_envs=5,
            train_env_class=DummyVectorEnv,
            test_env_class=DummyVectorEnv,
        )

        results = agent.train(verbose=False)
        self.assertIsInstance(results, dict)

        test_result = agent.test()
        self.assertIsInstance(test_result, dict)

class TestAgentDQN(TestAgentMixin, unittest.TestCase):
    TaskName = 'LunarLander-v2'
    AgentPreset = dqn_classic

class TestAgentSAC(TestAgentMixin, unittest.TestCase):
    TaskName = 'MountainCarContinuous-v0'
    AgentPreset = sac_classic
    
class TestAgentSACDictEnv(TestAgentMixin, unittest.TestCase):
    TaskName = 'PendulumDict-v0'
    AgentPreset = sac_classic
    