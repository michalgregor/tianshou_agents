import unittest

# suppress tqdm output
import os
from functools import partialmethod
from tqdm import tqdm
tqdm.__init__ = partialmethod(tqdm.__init__, file=open(os.devnull, 'w'))

from tianshou_agents.components import PassiveCollector
from tianshou_agents.methods.sac import sac_simple
from tianshou.env import DummyVectorEnv
from tianshou.data import Batch
import numpy as np

class TestPassiveCollector(unittest.TestCase):
    TaskName = ''
    AgentPreset = None
    Task = None

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testCollection(self):
        agent = sac_simple(
            'Pendulum-v1', stop_criterion=-250, seed=0, train_envs=5,
            train_env_class=DummyVectorEnv
        )

        passive_collector = PassiveCollector(
            policy=agent.policy,
            buffer=agent.train_collector.buffer,
            num_envs=len(agent.train_envs),
        )

        agent.train_collector.reset()
        passive_collector.make_collect(n_step=12)

        obs = agent.train_envs.reset()
        state = None
        done = None

        for istep in range(3):
            act_batch, state = passive_collector.compute_action(obs, state=state, done=done)
            ready_env_ids = passive_collector.ready_env_ids
            obs_next, rew, done, info = agent.train_envs.step(act_batch.act, ready_env_ids)

            transition = Batch(obs=obs, obs_next=obs_next, rew=rew, done=done, info=info)
            transition.update(**act_batch, **state)

            ret = passive_collector.observe_transition(transition)

            if istep < 2:
                self.assertIsNone(ret)
            else:
                self.assertIsNotNone(ret)
                break

            obs = obs_next

            ready_env_ids = passive_collector.ready_env_ids
            envs_done_local = np.where(done)[0]

            if len(envs_done_local) > 0:
                envs_done_global = ready_env_ids[envs_done_local]
                obs[envs_done_local] = agent.train_envs.reset(envs_done_global)

        len_buffer = len(passive_collector.buffer)
        self.assertEqual(len_buffer, 15)
    