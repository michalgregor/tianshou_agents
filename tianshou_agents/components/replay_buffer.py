from .component import Component
from tianshou.data import VectorReplayBuffer, ReplayBuffer
from typing import Union, Any, Optional
from numbers import Number
import torch

class BaseReplayBufferComponent(Component):
    """The base of replay buffer components.

    In order to subclass, implement a constructor that takes in the
    required arguments and constructs the replay buffer, assigning
    it to self.replay_buffer.
    """

class ReplayBufferComponent(BaseReplayBufferComponent):
    def __init__(self,
        agent: 'ComponentAgent',
        num_envs: Optional[int] = None,
        device: Optional[Union[str, int, torch.device]] = None,
        seed: Optional[int] = None,
        config_arg: Optional[Union[int, ReplayBuffer]] = None,
        component_class: Any = None,
        **kwargs
    ):
        super().__init__()

        self._state_objs.extend([
            'replay_buffer'
        ])

        if component_class is None:
            component_class = VectorReplayBuffer

        if not config_arg is None:
            if isinstance(config_arg, ReplayBuffer):
                self.replay_buffer = config_arg
            elif isinstance(config_arg, Number):
                size = config_arg

                if num_envs is None:
                    envs = agent.train_envs

                    if envs is None:
                        num_envs = 1
                    else:
                        num_envs = len(envs)

                self.replay_buffer = component_class(size, num_envs, **kwargs)
            else:
                raise ValueError(
                    f"config_arg must be of type ReplayBuffer or int, "
                    f"but is of type {type(config_arg)}"
                )
            
        else:
            self.replay_buffer = component_class(**kwargs)
