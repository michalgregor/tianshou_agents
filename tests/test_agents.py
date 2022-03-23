import unittest

from tianshou.env.venvs import DummyVectorEnv
from tianshou_agents.methods.dqn import dqn_classic
from tianshou_agents.methods.sac import sac_classic

# suppress tqdm output
import os
from functools import partialmethod
from tqdm import tqdm
tqdm.__init__ = partialmethod(tqdm.__init__, file=open(os.devnull, 'w'))

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
            self.TaskName, task=self.Task, max_epoch=2,
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
    