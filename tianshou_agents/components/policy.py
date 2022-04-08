from ..utils import StateDictObject
from torch.optim import Optimizer
from tianshou.policy import BasePolicy
from typing import Optional, Union, Callable
from gym.spaces import Tuple as GymTuple
from ..networks import RLNetwork
import torch
import gym

class PolicyComponent(StateDictObject):
    def __init__(self,
        agent: "Agent",
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        make_policy: Optional[Callable[..., BasePolicy]],
        reward_threshold: float = None,
        device: Union[str, int, torch.device] = "cpu",
        seed: int = None
    ):
        """
        To subclass this, you need to implement at least a constructor with
        the following arguments:
        * agent: Agent,
        * observation_space: gym.spaces.Space,
        * action_space: gym.spaces.Space,
        * make_policy: Callable[[], BasePolicy],
        * reward_threshold: Optional[float],
        * device: Union[str, int, torch.device],
        * seed: Optional[int],
        * any other arguments you want to pass to the policy.

        The constructor must create a policy object that inherits from
        tianhsou.policy.BasePolicy and assign it to self.policy.

        The make_policy argument is a callable used to construct the actual
        policy; this should be set to a default value by the derived policy,
        but allowed to be overridden by the user (e.g. to wrap a default
        DQNPolicy in an ICMPolicy etc.).

        This policy attribute is automatically made part of the state dict.
        """
        super().__init__()

        self._state_objs.extend([
            'policy'
        ])

        self._device = device
        self.observation_space = observation_space
        self.action_space = action_space
        self.make_policy = make_policy
        self.reward_threshold = reward_threshold

        if isinstance(self.observation_space, GymTuple):
            self.state_shape = (osp.shape or osp.n for osp in self.observation_space)
        else:
            self.state_shape = self.observation_space.shape or self.observation_space.n

        self.action_shape = self.action_space.shape or self.action_space.n

    def construct_rlnet(
        self, module, state_shape, action_shape, **kwargs
    ):
        if module is None:
            module = RLNetwork(
                state_shape, action_shape,
                device=self._device,
                **kwargs
            ).to(self._device)
        elif isinstance(module, torch.nn.Module):
            module = module.to(self._device)
        elif isinstance(module, dict):
            kwargs = kwargs.copy()
            kwargs.update(module)
            module = kwargs.pop("__type__", RLNetwork)
            module = module(
                state_shape, action_shape,
                device=self._device,
                **kwargs
            ).to(self._device)

        else:
            module = module(
                state_shape, action_shape,
                device=self._device,
                **kwargs
            ).to(self._device)

        return module

    def construct_optim(self, optim, model_params):
        if optim is None:
            optim = torch.optim.Adam(model_params)
        elif isinstance(optim, Optimizer):
            pass
        elif isinstance(optim, dict):
            kwargs = optim.copy()
            optim = kwargs.pop("__type__", torch.optim.Adam)
            optim = optim(model_params, **kwargs)
        else:
            optim = optim(model_params)

        return optim