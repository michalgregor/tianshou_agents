from .agent import AgentPreset, OffPolicyAgent
from .schedule import Schedule, ConstSchedule, ExponentialSchedule
from .callback import ScheduleCallback
from .network import MLP
from tianshou.policy import DQNPolicy
from tianshou.env import DummyVectorEnv
from typing import Any, Optional, Union, Callable, Dict
from torch.optim import Optimizer
from numbers import Number
import torch

class DQNAgent(OffPolicyAgent):    
    def __init__(
        self,
        task_name: str,
        eps_test: Union[float, Schedule, Callable[[int, int], Schedule]] = 0.01,
        eps_train: Union[float, Schedule, Callable[[int, int], Schedule]] =
            lambda max_epoch, step_per_epoch: ExponentialSchedule(
                max_epoch*step_per_epoch*0.5, 0.73, 0.1
            ),
        gamma: float = 0.99,
        target_update_freq: int = 0,
        estimation_step: int = 1,
        qnetwork: Optional[Union[torch.nn.Module, Callable[..., torch.nn.Module]]] = None,
        qnetwork_params: Optional[Dict[str, Any]] = None,
        optim: Optional[Union[Optimizer, Callable[..., Optimizer]]] = None,
        optim_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any
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
            qnetwork ([Union[torch.nn.Module, Callable[..., torch.nn.Module]]]):
                The torch Module to be used as the Q-Network. Can be either
                a torch Module or ``callable(state_shape, action_shape, device, **qnetwork_params)``
                that returns a torch Module. Defaults to None, which means an
                RLNetwork is going to be constructed.
            qnetwork_params (Dict[str, Any]): The parameters to be
                passed to the Q-Network (if a new one is to be constructed).
                Defaults to None, which means an empty dictionary.
            optim (Union[Optimizer, Callable[..., Optimizer]]): The optimizer
                to use for training the Q-Network. This can either be an
                Optimizer instance or ``callable(parameters, **kwargs)``.
                Defaults to None, which means the Adam optimizer is going
                to be constructed.
            optim_params (Dict[str, Any]): The parameters to be
                passed to the optimizer (if a new one is to be constructed).
                Defaults to None, which means an empty dictionary.

        For additional arguments that need to (or can optionally) be supplied
        as keyword arguments, see ``tianshou_agents.Agent`` and
        ``tianshou_agents.OffPolicyAgent``, or better still use and modify
        one of the provided presets.

        """
        policy_kwargs = locals().copy()
        del policy_kwargs['self']
        del policy_kwargs['__class__']
        del policy_kwargs['kwargs']
        del policy_kwargs['task_name']

        super().__init__(task_name=task_name, method_name='dqn',
                         **kwargs, **policy_kwargs)

    def _setup_policy(self, 
        is_double, eps_test, eps_train,
        gamma, target_update_freq,
        estimation_step, reward_normalization, qnetwork, qnetwork_params,
        optim, optim_params
    ):
        self.qnetwork = self.construct_rlnet(
            qnetwork, qnetwork_params,
            self.state_shape, self.action_shape
        )

        self.optim = self.construct_optim(
            optim, optim_params, self.qnetwork.parameters()
        )

        # the algo
        self.policy = DQNPolicy(
            self.qnetwork, self.optim, gamma, estimation_step,
            target_update_freq=target_update_freq,
            reward_normalization=reward_normalization,
            is_double=is_double
        )

        # eps schedules
        if isinstance(eps_train, Schedule):
            self.eps_train = eps_train
        elif isinstance(eps_train, Number):
            self.eps_train = ConstSchedule(eps_train)
        else:
            self.eps_train = eps_train(self.max_epoch, self.step_per_epoch)

        self.train_callbacks.append(ScheduleCallback(self.policy.set_eps, self.eps_train))

        if isinstance(eps_test, Schedule):
            self.eps_test = eps_test
        elif isinstance(eps_test, Number):
            self.eps_test = ConstSchedule(eps_test)
        else:
            self.eps_test = eps_test(self.max_epoch, self.step_per_epoch)

        self.test_callbacks.append(ScheduleCallback(self.policy.set_eps, self.eps_test))

# classic

dqn_classic_hyperparameters = {
    # dqn
    'is_double': True,
    'eps_test': 0.01,
    'eps_train': lambda max_epoch, step_per_epoch: ExponentialSchedule(
        max_epoch*step_per_epoch*0.5, 0.73, 0.1),
    'gamma': 0.99,
    'target_update_freq': 500,
    'estimation_step': 4,
    'reward_normalization': False,
    'qnetwork': None,
    'qnetwork_params': dict(
        model=MLP,
        hidden_sizes=[128, 128],
        dueling_param=(
            {"hidden_sizes": [128, 128]}, # Q_param
            {"hidden_sizes": [128, 128]} # V_param
        )
    ),
    'optim': None,
    'optim_params': dict(lr=0.013),
    # replay buffer
    'replay_buffer': 1000000,
    'prefill_steps': None,
    # general
    'train_envs': 16,
    'test_envs': 100,
    'train_env_class': DummyVectorEnv,
    'test_env_class': DummyVectorEnv,
    'episode_per_test': None,
    'seed': None,
    'max_epoch': 10,
    'step_per_epoch': 80000,
    'step_per_collect': None,
    'update_per_collect': 1.,
    'batch_size': 128,
    'logdir': 'log',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'train_callbacks': None,
    'test_callbacks': None,
    'train_collector': None,
    'test_collector': None,
    'exploration_noise_train': True,
    'exploration_noise_test': True,
    'task': None,
    'stop_criterion': True
}

dqn_classic = AgentPreset(DQNAgent, dqn_classic_hyperparameters)

# simple

dqn_simple_hyperparameters = {
    # dqn
    'is_double': True,
    'eps_test': 0.01,
    'eps_train': lambda max_epoch, step_per_epoch: ExponentialSchedule(
        max_epoch*step_per_epoch*0.5, 0.73, 0.1),
    'gamma': 0.99,
    'target_update_freq': 500,
    'estimation_step': 1,
    'reward_normalization': False,
    'qnetwork': None,
    'qnetwork_params': dict(
        model=MLP,
        hidden_sizes=[128, 128]
    ),
    'optim': None,
    'optim_params': dict(lr=0.013),
    # replay buffer
    'replay_buffer': 1000000,
    'prefill_steps': None,
    # general
    'train_envs': 16,
    'test_envs': 1,
    'train_env_class': DummyVectorEnv,
    'test_env_class': DummyVectorEnv,
    'episode_per_test': None,
    'seed': None,
    'max_epoch': 10,
    'step_per_epoch': 80000,
    'step_per_collect': None,
    'update_per_collect': 1.,
    'batch_size': 128,
    'logdir': 'log',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'train_callbacks': None,
    'test_callbacks': None,
    'train_collector': None,
    'test_collector': None,
    'exploration_noise_train': True,
    'exploration_noise_test': True,
    'task': None,
    'stop_criterion': True
}

dqn_simple = AgentPreset(DQNAgent, dqn_simple_hyperparameters)
