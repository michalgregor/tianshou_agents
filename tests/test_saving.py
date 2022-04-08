import torch
import unittest
from tianshou_agents.methods.sac import sac_simple
from tianshou_agents.callbacks import Callback

# suppress tqdm output
import os
from functools import partialmethod
from tqdm import tqdm
tqdm.__init__ = partialmethod(tqdm.__init__, file=open(os.devnull, 'w'))

class DummyStatefulCallback(Callback):
    def __init__(self, state=0):
        super().__init__()
        self.state = state
        self._state_objs.append('state')

    def __call__(self, epoch, env_step, gradient_step, agent):
        pass

class TestAgentCheckpointMixin:
    TaskName = ''
    AgentPreset = None
    Task = None

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testAgentTrainAndLoad(self):
        agent = self.AgentPreset(
            self.TaskName,
            task=self.Task,
            max_epoch=2,
            step_per_epoch=20,
            train_envs=1,
            test_envs=1,
            train_callbacks=[DummyStatefulCallback(5)]
        )

        results = agent.train(verbose=False)
        self.assertIsInstance(results, dict)
        epoch, env_step, gradient_step = agent.epoch, agent.env_step, agent.gradient_step
        log_path = agent.log_path

        agent = self.AgentPreset(
            self.TaskName, task=self.Task,
            max_epoch=2,
            step_per_epoch=20,
            train_envs=1,
            test_envs=1,
            train_callbacks=[DummyStatefulCallback(11)]
        )

        state_dict = torch.load(os.path.join(log_path, "last_agent.pth"))
        agent.load_state_dict(state_dict)

        self.assertEqual(agent.epoch, epoch)
        self.assertEqual(agent.env_step, env_step)
        self.assertEqual(agent.gradient_step, gradient_step)
        self.assertEqual(agent.train_callbacks[0].state, 5)

class TestCheckpointSAC(TestAgentCheckpointMixin, unittest.TestCase):
    TaskName = 'Pendulum-v0'
    AgentPreset = sac_simple