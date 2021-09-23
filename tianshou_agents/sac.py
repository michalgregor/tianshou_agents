from .agent import OffPolicyAgent
from .preset import AgentPreset
from .network import MLP
from tianshou.policy import SACPolicy
from tianshou.env import DummyVectorEnv
from tianshou.exploration import BaseNoise
from tianshou.utils.net.continuous import ActorProb, Critic
from typing import Any, Optional, Union, Callable, Dict
from torch.optim import Optimizer
import numpy as np
import torch

class SACAgent(OffPolicyAgent):
    def __init__(
        self,
        task_name: str,
        actor: Optional[Union[torch.nn.Module, Callable[..., torch.nn.Module], Dict[str, Any]]] = None,
        critic1: Optional[Union[torch.nn.Module, Callable[..., torch.nn.Module], Dict[str, Any]]] = None,
        critic2: Optional[Union[torch.nn.Module, Callable[..., torch.nn.Module], Dict[str, Any]]] = None,
        gamma: float = 0.99,
        tau: float =  0.005,
        alpha: float = 0.2,
        auto_alpha: bool = True,
        exploration_noise: Optional[BaseNoise] = None,
        reward_normalization: bool = False,
        estimation_step: int = 1,
        deterministic_eval: bool = True,
        actor_optim: Optional[Union[Optimizer, Callable[..., Optimizer], Dict[str, Any]]] = None,
        critic1_optim: Optional[Union[Optimizer, Callable[..., Optimizer], Dict[str, Any]]] = None,
        critic2_optim: Optional[Union[Optimizer, Callable[..., Optimizer], Dict[str, Any]]] = None,
        alpha_optim: Optional[Union[Optimizer, Callable[..., Optimizer], Dict[str, Any]]] = None,
        **kwargs
    ):
        """Implementation of Soft Actor-Critic. arXiv:1812.05905.

        Args:
            task_name (str): The name of the ``gym`` environment; by default,
                environments are constructed using ``gym.make``. To override
                this behaviour, supply a ``task`` argument: a callable that
                constructs your ``gym`` environment.
            actor (Union[torch.nn.Module, Callable[..., torch.nn.Module], Dict[str, Any]], optional):
                The torch Module to be used as the actor. Can be either
                a torch ``Module`` or ``callable(state_shape, action_shape, device)``
                that returns a torch ``Module``. If None, a default RLNetwork
                is constructed.
                
                Alternatively, this can be a dictionary, where the ``type`` key
                (RLNetwork by default) is a
                ``callable(state_shape, action_shape, device, **qnetwork_params)``
                and the remaining keys are ``**qnetwork_params``.

            critic1 (Union[torch.nn.Module, Callable[..., torch.nn.Module], Dict[str, Any]], optional):
                The torch Module to be used as the first critic. Can be either
                a torch ``Module`` or ``callable(state_shape, action_shape, device)``
                that returns a torch ``Module``. If None, a default RLNetwork
                is constructed.
                
                Alternatively, this can be a dictionary, where the ``type`` key
                (RLNetwork by default) is a
                ``callable(state_shape, action_shape, device, **qnetwork_params)``
                and the remaining keys are ``**qnetwork_params``.

            critic2 (Union[torch.nn.Module, Callable[..., torch.nn.Module], Dict[str, Any]], optional):
                The torch Module to be used as the second critic. Can be either
                a torch ``Module`` or ``callable(state_shape, action_shape, device)``
                that returns a torch ``Module``. If None, a default RLNetwork
                is constructed.
                
                Alternatively, this can be a dictionary, where the ``type`` key
                (RLNetwork by default) is a
                ``callable(state_shape, action_shape, device, **qnetwork_params)``
                and the remaining keys are ``**qnetwork_params``.

            gamma (float, optional): The discount rate.
            tau (float, optional): Param for the soft update of the target network.
            alpha (float, optional): [description]. The entropy regularization
                coefficient. Note: if ``auto_alpha`` is ``True``, this is
                tuned automatically during training.
            auto_alpha (bool, optional): Specifies whether alpha should be
                tuned automatically during training.
            exploration_noise (BaseNoise, optional):
                Noise added to actions for exploration. This is useful when
                solving hard-exploration problems. By default this is None,
                which means no noise is going to be used. If you use noise,
                you need to ensure that its scale w.r.t. the scale of the
                actions is appropriate.
            reward_normalization (bool, optional): Normalize the reward to Normal(0, 1).
            estimation_step (int, optional): The number of steps to look ahead.
            deterministic_eval (bool, optional): Whether to use deterministic
                action (mean of Gaussian policy) instead of stochastic action
                sampled by the policy.
            actor_optim (Union[Optimizer, Callable[..., Optimizer], Dict[str, Any]], optional):
                The optimizer to use for training the actor. This can either be
                an Optimizer instance or ``callable(parameters)``.
                Defaults to None, which means the Adam optimizer is going
                to be constructed.

                Alternatively, this can be a dictionary, where the ``type`` key
                (Adam) is a ``callable(parameters, **kwargs)`` and the 
                remaining keys are ``**kwargs``.
            critic1_optim (Union[Optimizer, Callable[..., Optimizer], Dict[str, Any]], optional):
                The optimizer to use for training the first critic. This can
                either be an Optimizer instance or ``callable(parameters)``.
                Defaults to None, which means the Adam optimizer is going
                to be constructed.

                Alternatively, this can be a dictionary, where the ``type`` key
                (Adam) is a ``callable(parameters, **kwargs)`` and the 
                remaining keys are ``**kwargs``.
            critic2_optim (Union[Optimizer, Callable[..., Optimizer], Dict[str, Any]], optional):
                The optimizer to use for training the second critic. This can
                either be an Optimizer instance or ``callable(parameters)``.
                Defaults to None, which means the Adam optimizer is going
                to be constructed.

                Alternatively, this can be a dictionary, where the ``type`` key
                (Adam) is a ``callable(parameters, **kwargs)`` and the 
                remaining keys are ``**kwargs``.
            alpha_optim (Union[Optimizer, Callable[..., Optimizer], Dict[str, Any]], optional):
                The optimizer to use for tuning alpha. This can either be
                an Optimizer instance or ``callable(parameters)``.
                Defaults to None, which means the Adam optimizer is going
                to be constructed.

                Alternatively, this can be a dictionary, where the ``type`` key
                (Adam) is a ``callable(parameters, **kwargs)`` and the 
                remaining keys are ``**kwargs``.

                Note that this optimizer typically needs to have a learning
                rate lower than the other optimizers.

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

        super().__init__(task_name=task_name, method_name='sac',
                         **kwargs, **policy_kwargs)

    def _setup_policy(self,
        actor,
        critic1,
        critic2,
        gamma, tau,
        auto_alpha, alpha,
        exploration_noise,
        reward_normalization,
        estimation_step,
        deterministic_eval,
        actor_optim,
        critic1_optim,
        critic2_optim,
        alpha_optim,
    ):
        max_action = np.max(self.action_space.high)

        # actor
        self.actor_net = self.construct_rlnet(
            actor, self.state_shape, 0
        )

        self.actor = ActorProb(
            self.actor_net, self.action_shape,
            max_action=max_action, device=self._device, unbounded=True
        ).to(self._device)
        self.actor_optim = self.construct_optim(
            actor_optim, self.actor.parameters()
        )

        # critic 1
        self.critic1_net = self.construct_rlnet(
            critic1, self.state_shape,
            self.action_shape, actions_as_input=True
        )

        self.critic1 = Critic(
            self.critic1_net, device=self._device).to(self._device)
        self.critic1_optim = self.construct_optim(
            critic1_optim, self.critic1.parameters()
        )

        # critic 2
        self.critic2_net = self.construct_rlnet(
            critic2, self.state_shape,
            self.action_shape, actions_as_input=True
        )

        self.critic2 = Critic(
            self.critic2_net, device=self._device).to(self._device)
        self.critic2_optim = self.construct_optim(
            critic2_optim, self.critic2.parameters()
        )

        # alpha tuning
        if auto_alpha:
            target_entropy = -np.prod(self.action_space.shape)
            log_alpha = torch.zeros(1, requires_grad=True, device=self._device)
            alpha_optim = self.construct_optim(
                alpha_optim, [log_alpha]
            )
            alpha = (target_entropy, log_alpha, alpha_optim)

        self.policy = SACPolicy(
            self.actor, self.actor_optim,
            self.critic1, self.critic1_optim,
            self.critic2, self.critic2_optim,
            tau=tau, gamma=gamma, alpha=alpha,
            reward_normalization=reward_normalization,
            exploration_noise=exploration_noise,
            action_space=self.action_space,
            estimation_step=estimation_step,
            deterministic_eval=deterministic_eval
        )

# the simple preset

sac_simple_hyperparameters = {
    # sac
    'exploration_noise': None,
    'gamma': 0.99,
    'tau': 0.005,
    'auto_alpha': True,
    'alpha': 0.2,
    'reward_normalization': False,
    'estimation_step': 1,
    'deterministic_eval': True,
    'actor': dict(model=MLP, hidden_sizes=[128, 128]),
    'actor_optim': dict(lr=3e-4),
    'critic1': dict(model=MLP, hidden_sizes=[128, 128]),
    'critic1_optim': dict(lr=3e-4),
    'critic2': dict(model=MLP, hidden_sizes=[128, 128]),
    'critic2_optim': dict(lr=3e-4),
    'alpha_optim': dict(lr=3e-4),
    # replay buffer
    'replay_buffer': 1000000,
    'prefill_steps': None,
    # general
    'train_envs': 1,
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

sac_simple = AgentPreset(SACAgent, sac_simple_hyperparameters)

# the classic preset

sac_classic_hyperparameters = {
    # sac
    'exploration_noise': None,
    'gamma': 0.99,
    'tau': 0.005,
    'auto_alpha': True,
    'alpha': 0.2,
    'reward_normalization': False,
    'estimation_step': 1,
    'deterministic_eval': True,
    'actor': dict(model=MLP, hidden_sizes=[128, 128]),
    'actor_optim': dict(lr=3e-4),
    'critic1': dict(model=MLP, hidden_sizes=[128, 128]),
    'critic1_optim': dict(lr=3e-4),
    'critic2': dict(model=MLP, hidden_sizes=[128, 128]),
    'critic2_optim': dict(lr=3e-4),
    'alpha_optim': dict(lr=3e-4),
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

sac_classic = AgentPreset(SACAgent, sac_classic_hyperparameters)

# the pybullet preset

sac_pybullet_hyperparameters = {
    # sac
    'exploration_noise': None,
    'gamma': 0.99,
    'tau': 0.005,
    'auto_alpha': True,
    'alpha': 0.2,
    'reward_normalization': False,
    'estimation_step': 1,
    'deterministic_eval': True,
    'actor': dict(model=MLP, hidden_sizes=[256, 256]),
    'actor_optim': dict(lr=1e-3),
    'critic1': dict(model=MLP, hidden_sizes=[256, 256]),
    'critic1_optim': dict(lr=1e-3),
    'critic2': dict(model=MLP, hidden_sizes=[256, 256]),
    'critic2_optim': dict(lr=1e-3),
    'alpha_optim': dict(lr=3e-4),
    # replay buffer
    'replay_buffer': 1000000,
    'prefill_steps': 10000,
    # general
    'train_envs': 1,
    'test_envs': 10,
    'train_env_class': DummyVectorEnv,
    'test_env_class': DummyVectorEnv,
    'episode_per_test': None,
    'seed': None,
    'max_epoch': 200,
    'step_per_epoch': 5000,
    'step_per_collect': None,
    'update_per_collect': 1.,
    'batch_size': 256,
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

sac_pybullet = AgentPreset(SACAgent, sac_pybullet_hyperparameters)
