from ..utils import ConfigBuilder
from .component import Component
from typing import Union, List, Tuple, Sequence
from ..networks import RLNetwork
import torch

class BasePolicyComponent(Component):
    """The base of policy components.

    In order to subclass, implement a constructor that takes in the
    required arguments and constructs the policy, assigning
    it to self.policy.

    Most policies will need to construct models and optimizers. This can be
    done using construct_optim and construct_rlnet, which both internally
    rely on ConfigBuilder.

    In construct_rlnet, an RLNetwork is constructed by default; also consult
    its documentation to see how it can be used to wrap an existing nn.Module
    to make it compatible with the tianshou.RL interface.

    Args:
        method_name: The name of the method in a format without special
            characters, so that it can be used as part of a path specification,
            etc. – e.g. 'dqn' or 'a2c'.
    """

    module_builder = ConfigBuilder(
        torch.nn.Module,
        default_obj_constructor=RLNetwork,
    )

    optim_builder = ConfigBuilder(
        torch.optim.Optimizer,
            default_obj_constructor=torch.optim.Adam,
    )

    def __init__(self, method_name: str, **kwargs):
        super().__init__(**kwargs)
        self.method_name = method_name

    def construct_rlnet(
        self,
        module,
        observation_shape: Union[int, Tuple[int], List[Tuple[int]]],
        action_shape: Union[int, Sequence[int]],
        device: Union[str, int, torch.device],
        **model_kwargs
    ):
        module = self.module_builder(
            config=module,
            default_kwargs=dict(model_kwargs,
                observation_shape=observation_shape,
                action_shape=action_shape,
                device=device
            )
        )
        
        if not module is None:
            module = module.to(device)
        
        return module

    def construct_optim(
        self, optim, model_params, **kwargs
    ):
        return self.optim_builder(
            config=optim,
            default_kwargs=dict(kwargs,
                params=model_params
            )
        )
