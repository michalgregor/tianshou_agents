from ..agent import ComponentAgent
from ..components.preset import AgentPreset
from ..components import BasePolicyComponent
from tianshou.policy import BasePolicy, MultiAgentPolicyManager
from typing import Any, Optional, Union, Callable, List, Dict
import torch

try:
    from tianshou.env.pettingzoo_env import PettingZooEnv
except ImportError:
    PettingZooEnv = None  # type: ignore

class MultiAgentPolicyManagerComponent(BasePolicyComponent):
    def __init__(
        self,
        # component args
        agent: ComponentAgent,
        device: Optional[Union[str, int, torch.device]] = None,
        seed: Optional[int] = None,
        # context args
        config_arg: Optional[List[Union[
            BasePolicy,
            BasePolicyComponent,
            Callable[..., BasePolicyComponent],
            Dict[str, Any]
        ]]] = None,
        component_class: Any = None,
        # method args
        policies: Optional[Union[
            List[Union[
                BasePolicy,
                BasePolicyComponent,
                Callable[..., BasePolicyComponent],
                Dict[str, Any]
            ]],
            Callable[..., List[BasePolicyComponent]]
        ]] = None,
        env: PettingZooEnv = None,
        ma_kwargs = None,
        method_name: str = "ma",
        **policy_kwargs
    ):
        super().__init__(method_name=method_name)

        # the state dict
        self._state_objs.extend([
            'policies'
        ])

        if policies is None:
            policies = config_arg

        assert policies is not None, "either policies or config_arg must be provided"

        if isinstance(policies, list):
            self.policy_components = [
                agent.config_router.policy_builder(
                    config=component_policy,
                    default_kwargs=dict(policy_kwargs,
                        agent=agent,
                        device=device,
                        seed=seed
                    )
                ) for component_policy in policies
            ]
        elif callable(policies):
            self.policy_components = policies(
                **dict(policy_kwargs,
                    agent=agent,
                    device=device,
                    seed=seed
                )
            )
        else:
            raise ValueError("policies must be a list or a callable")

        if component_class is None:
            component_class = MultiAgentPolicyManager

        if ma_kwargs is None:
            ma_kwargs = {}

        if env is None and not agent.train_envs is None:
            env = agent.train_envs.workers[0].env

        assert env is not None, "a PettingZooEnv must be provided"

        self.policy = component_class(
            policies=self.policies,
            env=env,
            **ma_kwargs
        )

    @property
    def policies(self):
        return [policy_component.policy for policy_component in self.policy_components]

    def setup(
        self,
        agent: 'ComponentAgent',
        device: Optional[Union[str, int, torch.device]] = None,
        seed: Optional[int] = None
    ):
        for policy_component in self.policy_components:
            policy_component.setup(
                agent=agent,
                device=device,
                seed=seed
            )

# base config

ma_base_config = {
    # agent
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': None,
    # replay buffer
    'replay_buffer': 1000000,
    # train collector
    'train_collector': {},
    'train_envs': 1,
    'train_env_class': None,
    'exploration_noise_train': True,
    # test collector
    'test_collector': {},
    'test_envs': 1,
    'test_env_class': None,
    'exploration_noise_test': True,
    # policy    
    'policy': MultiAgentPolicyManagerComponent,
    'policies': None, # this needs to be supplied upon 
    'env': None,
    'ma_kwargs': None,
    # trainer
    'trainer': {},
    'trainer_class': None, # this needs to be supplied upon construction
    'max_epoch': 10,
    'step_per_epoch': 80000,
    'prefill_steps': None,
    'episode_per_test': None,
    'step_per_collect': None,
    'update_per_collect': 1.,
    'batch_size': 128,
    'train_callbacks': None,
    'test_callbacks': None,
    'stop_criterion': None,
    'save_best_callbacks': {},
    'save_checkpoint_callbacks': {},
    # logger
    'logger': 'log',
}

ma_default = AgentPreset(ma_base_config)
