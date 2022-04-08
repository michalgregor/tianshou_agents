from ..agent import Agent, OffPolicyAgent
from ..components.preset import AgentPreset
from ..schedules import Schedule, ConstSchedule, ExponentialSchedule
from ..callbacks import ScheduleCallback
from ..networks import MLP
from ..components import PolicyComponent
from ..utils import derive_conf
from tianshou.policy import DQNPolicy, BasePolicy
from typing import Any, Optional, Union, Callable, Dict
from torch.optim import Optimizer
from numbers import Number
import torch
import gym

class DQNPolicyComponent(PolicyComponent):
    def __init__(
        self,
        agent: Agent,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        reward_threshold: float = None,
        device: Union[str, int, torch.device] = "cpu",
        seed: int = None,
        is_double: bool = True,
        eps_test: Union[float, Schedule, Callable[[int, int], Schedule]] = 0.01,
        eps_train: Union[float, Schedule, Callable[[int, int], Schedule]] =
            lambda max_epoch, step_per_epoch: ExponentialSchedule(
                max_epoch*step_per_epoch*0.5, 0.73, 0.1
            ),
        gamma: float = 0.99,
        target_update_freq: int = 0,
        estimation_step: int = 1,
        reward_normalization: bool = False,
        qnetwork: Optional[Union[torch.nn.Module, Callable[..., torch.nn.Module], Dict[str, Any]]] = None,
        optim: Optional[Union[Optimizer, Callable[..., Optimizer], Dict[str, Any]]] = None,
        make_policy: Optional[Callable[..., BasePolicy]] = DQNPolicy,
        **policy_kwargs: Any
    ):
        """Implementation of Deep Q Network. arXiv:1312.5602.

        Implementation of Double Q-Learning. arXiv:1509.06461.

        Implementation of Dueling DQN. arXiv:1511.06581 (the dueling DQN is
        implemented in the network side, not here).

        Args:
            task_name (str): The name of the ``gym`` environment; by default,
                environments are constructed using ``gym.make``. To override
                this behaviour, supply a ``task`` argument: a callable that
                constructs your ``gym`` environment.
            eps_test (Union[float, Schedule, Callable[[int, int], Schedule]]):
                The test-time epsilon (exploration rate); can be a float,
                a schedule or a ``callable(max_epoch, step_per_epoch)``
                that constructs a schedule.
            eps_train (Union[float, Schedule, Callable[[int, int], Schedule]]):
                The train-time epsilon (exploration rate); can be a float,
                a schedule or a ``callable(max_epoch, step_per_epoch)``
                that constructs a schedule.
            gamma (float): The discount rate.
            target_update_freq (int): The target network update frequency
                (0 if you do not use the target network).
            estimation_step (int): The number of steps to look ahead.
            qnetwork (Union[torch.nn.Module, Callable[..., torch.nn.Module], Dict[str, Any]], optional):
                The torch Module to be used as the Q-Network. Can be either
                a torch ``Module`` or ``callable(state_shape, action_shape, device)``
                that returns a torch ``Module``. If None, a default RLNetwork
                is constructed.
                
                Alternatively, this can be a dictionary, where the ``type`` key
                (RLNetwork by default) is a
                ``callable(state_shape, action_shape, device, **qnetwork_params)``
                and the remaining keys are ``**qnetwork_params``.

            optim (Union[Optimizer, Callable[..., Optimizer], Dict[str, Any]], optional):
                The optimizer to use for training the Q-Network. This can either be an
                Optimizer instance or ``callable(parameters)``.
                Defaults to None, which means the Adam optimizer is going
                to be constructed.

                Alternatively, this can be a dictionary, where the ``type`` key
                (Adam) is a ``callable(parameters, **kwargs)`` and the 
                remaining keys are ``**kwargs``.

            make_policy (Callable[..., BasePolicy], optional):
                A callable that constructs the policy. Defaults to DQNPolicy;
                the arguments are the same as the arguments of DQNPolicy.
                The make_policy argument can be used to add custom wrappers
                to the base policy, etc.

        For additional arguments that need to (or can optionally) be supplied
        as keyword arguments, see ``tianshou_agents.Agent`` and
        ``tianshou_agents.OffPolicyAgent``, or better still use and modify
        one of the provided presets.

        """
        super().__init__(
            agent=agent,
            observation_space=observation_space,
            action_space=action_space,
            reward_threshold=reward_threshold,
            device=device,
            seed=seed,
            make_policy=make_policy
        )

        # the state dict
        self._state_objs.append('optim')

        # the network
        self.qnetwork = self.construct_rlnet(
            qnetwork, self.state_shape, self.action_shape
        )

        self.optim = self.construct_optim(
            optim, self.qnetwork.parameters()
        )

        # the algo
        self.policy = self.make_policy(
            self.qnetwork, self.optim, gamma, estimation_step,
            target_update_freq=target_update_freq,
            reward_normalization=reward_normalization,
            is_double=is_double,
            **policy_kwargs
        )

        # eps schedules
        if isinstance(eps_train, Schedule):
            eps_train = eps_train
        elif isinstance(eps_train, Number):
            eps_train = ConstSchedule(eps_train)
        else:
            eps_train = eps_train(agent.max_epoch, agent.step_per_epoch)

        agent.train_callbacks.append(ScheduleCallback(self.policy.set_eps, eps_train))

        if isinstance(eps_test, Schedule):
            eps_test = eps_test
        elif isinstance(eps_test, Number):
            eps_test = ConstSchedule(eps_test)
        else:
            eps_test = eps_test(agent.max_epoch, agent.step_per_epoch)

        agent.test_callbacks.append(ScheduleCallback(self.policy.set_eps, eps_test))

# base config

dqn_base_config = {
    # agent
    'policy_component': DQNPolicyComponent,
    'make_policy': DQNPolicy,
    # dqn
    'is_double': True,
    'eps_test': 0.01,
    'eps_train': lambda max_epoch, step_per_epoch: ExponentialSchedule(
        max_epoch*step_per_epoch*0.5, 0.73, 0.1),
    'gamma': 0.99,
    'target_update_freq': 500,
    'estimation_step': 1,
    'reward_normalization': False,
    'qnetwork': dict(
        model=MLP,
        hidden_sizes=[128, 128],
        dueling_param=(
            {"hidden_sizes": [128, 128]}, # Q_param
            {"hidden_sizes": [128, 128]} # V_param
        )
    ),
    'optim': dict(lr=0.013),
    # replay buffer
    'replay_buffer': 1000000,
    'prefill_steps': None,
    # general
    'train_envs': 1,
    'test_envs': 1,
    'train_env_class': None,
    'test_env_class': None,
    'episode_per_test': None,
    'seed': None,
    'max_epoch': 10,
    'step_per_epoch': 80000,
    'step_per_collect': None,
    'update_per_collect': 1.,
    'batch_size': 128,
    'logger': 'log',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'train_callbacks': None,
    'test_callbacks': None,
    'train_collector': None,
    'test_collector': None,
    'exploration_noise_train': True,
    'exploration_noise_test': True,
    'task': None,
    'stop_criterion': False
}

dqn_default = AgentPreset(OffPolicyAgent, dqn_base_config)

# classic

dqn_classic_hyperparameters = derive_conf(dqn_base_config, {
    # general
    'train_envs': 16,
    'test_envs': 100,
})

dqn_classic = AgentPreset(OffPolicyAgent, dqn_classic_hyperparameters)

# simple

dqn_simple_hyperparameters = derive_conf(dqn_base_config, {
    # dqn
    'qnetwork': dict(
        model=MLP,
        hidden_sizes=[128, 128]
    ),
    # general
    'train_envs': 1,
    'test_envs': 5,
    'episode_per_test': 10,
})

dqn_simple = AgentPreset(OffPolicyAgent, dqn_simple_hyperparameters)
