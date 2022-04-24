import unittest

from tianshou_agents.methods.dqn import dqn_simple
from tianshou.data import Batch
from tianshou.env import DummyVectorEnv
import numpy as np
import gym

# suppress tqdm output
import os
from functools import partialmethod
from tqdm import tqdm
tqdm.__init__ = partialmethod(tqdm.__init__, file=open(os.devnull, 'w'))

class TestComponentAgent(unittest.TestCase):
    def setUp(self):
        self.env = DummyVectorEnv([lambda: gym.make('LunarLander-v2')])

        self.agent = dqn_simple(
            'LunarLander-v2', seed=1,
            component_train_collector=None,
            component_test_collector=None,
            observation_space=self.env.observation_space[0],
            action_space=self.env.action_space[0],
        )

    def tearDown(self):
        pass

    def test_component_agent_construct(self):
        env = self.env
        agent = self.agent
        agent.init_passive_training(step_per_epoch=10)

        state = None
        done = None
        obs = env.reset()

        for istep in range(15):
            act_batch, state = agent.act(obs, state=state, done=done)
            obs_next, rew, done, info = env.step(act_batch.act)

            transition = Batch(obs=obs, obs_next=obs_next, rew=rew, done=done, info=info)
            transition.update(**act_batch, **state)

            agent.observe_transition(transition)
            obs = obs_next
            done_envs = np.where(done)[0]

            if len(done_envs):
                env.reset(done_envs)   

        self.assertEqual(istep, 14)
        self.assertEqual(len(agent.buffer), 15)
