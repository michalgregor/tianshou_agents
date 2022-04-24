import unittest

from tianshou_agents.components import (
    CollectorComponent, ReplayBufferComponent,
    LoggerComponent, TrainerComponent
)
from tianshou_agents.agent import ComponentAgent
from tianshou_agents.methods.dqn import DQNPolicyComponent
from tianshou_agents.networks import MLP
from tianshou.trainer import OffpolicyTrainer
from tianshou.policy import DQNPolicy

# suppress tqdm output
import os
from functools import partialmethod
from tqdm import tqdm
tqdm.__init__ = partialmethod(tqdm.__init__, file=open(os.devnull, 'w'))

class TestComponentAgent(unittest.TestCase):
    def setUp(self):
        self.agent = ComponentAgent(
            component_replay_buffer=10,
            component_train_collector={
                'envs': 1,
                'task_name': 'LunarLander-v2'
            },
            component_test_collector={
                'envs': 1,
                'task_name': 'LunarLander-v2'
            },
            component_policy={
                '__type__': DQNPolicyComponent,
                'qnetwork': dict(
                    model=MLP,
                    hidden_sizes=[16, 16],
                    dueling_param=(
                        {"hidden_sizes": [16, 16]}, # Q_param
                        {"hidden_sizes": [16, 16]} # V_param
                    )
                ),
                'max_epoch': 2,
                'step_per_epoch': 3,
            },
            component_logger='log',
            component_trainer={
                'component_class': OffpolicyTrainer,
                'max_epoch': 2,
                'step_per_epoch': 3,
            },
            device=None,
            seed=None
        )

    def tearDown(self):
        pass

    def test_component_agent_construct(self):
        self.assertIsInstance(
            self.agent.component_train_collector,
            CollectorComponent
        )

        self.assertIsInstance(
            self.agent.component_test_collector,
            CollectorComponent
        )

        self.assertIsInstance(
            self.agent.component_policy,
            DQNPolicyComponent
        )

        self.assertIsInstance(
            self.agent.policy,
            DQNPolicy
        )

        self.assertIsInstance(
            self.agent.component_replay_buffer,
            ReplayBufferComponent
        )

        self.assertIsInstance(
            self.agent.component_logger,
            LoggerComponent
        )

        self.assertIsInstance(
            self.agent.logger,
            LoggerComponent
        )

        self.assertIsInstance(
            self.agent.component_trainer,
            TrainerComponent
        )

    def test_component_agent_train(self):
        self.agent.train()
        self.assertEqual(self.agent.epoch, 2)

        self.agent.train()
        self.assertEqual(self.agent.epoch, 2)

        self.agent.train(max_epoch=3)
        self.assertEqual(self.agent.epoch, 3)

    def test_make_trainer(self):
        trainer = self.agent.make_trainer()
        trainer.run()
        self.assertEqual(self.agent.epoch, 2)

        trainer = self.agent.make_trainer(max_epoch=2)
        trainer.run()
        self.assertEqual(self.agent.epoch, 2)

        trainer = self.agent.make_trainer(max_epoch=3)
        trainer.run()
        self.assertEqual(self.agent.epoch, 3)

    def test_independent_trainers(self):
        trainer = self.agent.make_trainer(resume_from_log=False)
        trainer.run()
        self.assertEqual(self.agent.epoch, 2)
        self.assertEqual(trainer.epoch, 3) # iteration stops at max_epoch+1

        trainer = self.agent.make_trainer(max_epoch=2, resume_from_log=False)
        trainer.run()
        self.assertEqual(self.agent.epoch, 4)
        self.assertEqual(trainer.epoch, 3) # iteration stops at max_epoch+1

        trainer = self.agent.make_trainer(max_epoch=3, resume_from_log=False)
        trainer.run()
        self.assertEqual(self.agent.epoch, 7)
        self.assertEqual(trainer.epoch, 4) # iteration stops at max_epoch+1

    def test_interleaved_trainers(self):
        trainer1 = self.agent.make_trainer(resume_from_log=False)
        trainer2 = self.agent.make_trainer(resume_from_log=False)

        for result1, result2 in zip(trainer1, trainer2):
            pass
        
        self.assertEqual(self.agent.epoch, 4)
        self.assertEqual(trainer1.epoch, 3) # iteration stops at max_epoch+1
        self.assertEqual(trainer2.epoch, 2) # zip iteration stops with trainer1
