from .env import setup_envs
from .component import Component
from .replay_buffer import BaseReplayBufferComponent
from tianshou.data import Collector, ReplayBuffer
from tianshou.env import BaseVectorEnv
from tianshou.policy import BasePolicy, RandomPolicy
from functools import partial
from typing import Optional, Callable, Dict, Any, Union, List, Type
import torch
import gym

class BaseCollectorComponent(Component):
    """The base of collector components.

    In order to subclass, implement a constructor that takes in the
    required arguments and constructs the collector, assigning
    it to self.collector.
    """

class CollectorComponent(BaseCollectorComponent):
    def __init__(self,
        agent: 'ComponentAgent',
        device: Optional[Union[str, int, torch.device]] = None,
        seed: Optional[int] = None,
        replay_buffer: Optional[Union[
            int,
            ReplayBuffer,
            BaseReplayBufferComponent,
            Callable[..., BaseReplayBufferComponent],
            Dict[str, Any]
        ]] = None,
        task: Optional[Callable[[], gym.Env]] = None,
        task_name: Optional[str] = None,
        env: Optional[Union[int, List[Union[gym.Env, Callable[[], gym.Env]]], BaseVectorEnv]] = None,
        envs: Optional[Union[int, List[Union[gym.Env, Callable[[], gym.Env]]], BaseVectorEnv]] = None,
        env_class: Optional[Type[BaseVectorEnv]] = None,
        policy: Optional[BasePolicy] = None,
        config_arg: Optional[Union[Collector, int]] = None,
        component_class: Any = None,
        **kwargs
    ):
        super().__init__()
        self.task_name = task_name

        self._state_objs.extend([
            'collector.buffer'
        ])

        # envs is an alias of env for backward compatibility
        if env is None: env = envs

        # setup the default collector class
        if component_class is None:
            component_class = Collector

        # if there is a pre-constructed collector, use it
        if isinstance(config_arg, Collector):
            self.collector = config_arg
        else:
            if task is None:
                assert not task_name is None
                task = partial(gym.make, task_name)

            # construct the enviroments if necessary
            env = setup_envs(task, env_class, env, seed=seed)

            if policy is None:
                policy = RandomPolicy(env.action_space[0])

            # constructs the replay buffer component if necessary
            if agent.component_replay_buffer is None:
                agent.component_replay_buffer = agent.config_router.replay_buffer_builder(
                    config=replay_buffer,
                    default_kwargs=dict(
                        agent=agent,
                        device=device,
                        seed=seed,
                        num_envs=len(env)
                    )
                )

            # construct the actual collector
            self.collector = component_class(
                policy=policy,
                env=env,
                # note: buffer is None for the test collector
                buffer=agent.component_replay_buffer.replay_buffer
                    if not replay_buffer is None else None,
                **kwargs
            )

    def setup(
        self,
        agent: 'ComponentAgent',
        device: Optional[Union[str, int, torch.device]] = None,
        seed: Optional[int] = None
    ):
        self.collector.policy = agent.policy

    @property
    def observation_space(self):
        return None if self.collector.env is None else self.collector.env.observation_space[0]

    @property
    def action_space(self):
        return None if self.collector.env is None else self.collector.env.action_space[0]

class DummyCollector:
    def __init__(self, buffer):
        self.buffer = buffer
