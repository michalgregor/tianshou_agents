from tianshou.data import Collector, VectorReplayBuffer, ReplayBuffer
from tianshou.policy import RandomPolicy, BasePolicy
from tianshou.env import BaseVectorEnv, DummyVectorEnv, SubprocVectorEnv
from .utils import StateDictObject, LoggerWrapper
from tianshou.utils.logger.base import BaseLogger
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Union, Dict, Any
from numbers import Number
import gym
import os

def setup_envs(task, env_class, envs):
    if isinstance(envs, int):
        if env_class is None:
            env_class = DummyVectorEnv if envs == 1 else SubprocVectorEnv

        envs = env_class(
            [task for _ in range(envs)]
        )
    elif isinstance(envs, list):
        if env_class is None:
            env_class = DummyVectorEnv if len(envs) == 1 else SubprocVectorEnv

        envs = env_class([lambda: env if isinstance(env, gym.Env) else env for env in envs])
    elif isinstance(envs, BaseVectorEnv):
        pass
    else:
        raise TypeError(f"envs: a BaseVectorEnv or an integer expected, got '{envs}'.")

    return envs

class AgentPolicy(StateDictObject):
    def __init__(self, policy):
        self.policy = policy

        self._state_objs.extend([
            'policy'
        ])

    @property
    def unwrapped(self):
        return self.policy

    @unwrapped.setter
    def unwrapped(self, policy):
        self.policy = policy

class AgentCollector(StateDictObject):
    def __init__(self,
        collector, task, env_class, envs, exploration_noise,
        replay_buffer = None, seed: int = None, **kwargs
    ):  
        super().__init__(**kwargs)
        self._state_objs.extend([
            'collector.buffer'
        ])

        if isinstance(collector, Collector):
            self.collector = collector
        else:
            envs = setup_envs(task, env_class, envs)
            placeholder_policy = RandomPolicy(envs.action_space[0])

            if collector is None: collector = Collector
            replay_buffer = self._create_replay_buffer(replay_buffer, envs)

            self.collector = collector(
                policy=placeholder_policy,
                env=envs,
                buffer=replay_buffer,
                exploration_noise=exploration_noise
            )

        if not seed is None:
            self.collector.env.seed(seed)

    @property
    def unwrapped(self):
        return self.collector

    @unwrapped.setter
    def unwrapped(self, value):
        self.collector = value
        
    def _create_replay_buffer(self, replay_buffer, envs):
        if replay_buffer is None:
            return None
        elif isinstance(replay_buffer, Number):
            return VectorReplayBuffer(replay_buffer, len(envs))
        elif isinstance(replay_buffer, ReplayBuffer):
            return replay_buffer
        else:
            return replay_buffer(len(envs))

class AgentLogger(StateDictObject, LoggerWrapper):
    def __init__(self,
        logger: Optional[Union[str, Dict[str, Any], BaseLogger]],
        task_name: str, method_name: str,
        seed: int = None
    ):
        self.env_step = None
        self.epoch = None
        self.gradient_step = None
        self.log_path = None

        if isinstance(logger, BaseLogger):
            pass
        elif logger is None:
            self.log_path = os.path.join("log", task_name, method_name)
            writer = SummaryWriter(self.log_path)
            logger = TensorboardLogger(writer)
        elif isinstance(logger, str):
            self.log_path = os.path.join(logger, task_name, method_name)
            writer = SummaryWriter(self.log_path)
            logger = TensorboardLogger(writer)
        else:
            logger_params = logger.copy()
            make_logger = logger_params.pop("type", TensorboardLogger)

            if make_logger == TensorboardLogger:
                writer = logger_params.get("writer")

                if writer is None:
                    log_path = logger_params.pop("log_path")
                    log_dir = logger_params.pop("log_dir", "log")

                    if not log_path is None:
                        self.log_path = log_path
                    else:
                        self.log_path = os.path.join(log_dir, task_name, method_name)

                    logger_params['writer'] = SummaryWriter(self.log_path)

                else:
                    self.log_path = writer.log_dir

            logger = make_logger(**logger_params)

        super().__init__(logger=logger)

        self._state_objs.extend([
            'epoch',
            'env_step',
            'gradient_step'
        ])

    def reset_progress_counters(self):
        self.env_step = 0
        self.epoch = 0
        self.gradient_step = 0
