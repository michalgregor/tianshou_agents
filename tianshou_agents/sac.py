from .agent import AgentPreset, OffPolicyAgent
from .network import MLP
from tianshou.policy import SACPolicy
from tianshou.env import DummyVectorEnv
from tianshou.exploration import OUNoise
from tianshou.utils.net.continuous import ActorProb, Critic
from typing import Any, Optional, Union, Callable, Dict
from torch.optim import Optimizer
import numpy as np
import torch

class SACAgent(OffPolicyAgent):
    def __init__(
        self,
        task_name: str,
        actor: Optional[torch.nn.Module] = None,
        actor_params: Optional[dict] = None,
        critic1: Optional[torch.nn.Module] = None,
        critic1_params: Optional[dict] = None,
        critic2: Optional[torch.nn.Module] = None,
        critic2_params: Optional[dict] = None,
        gamma: float = 0.99,
        tau: float =  0.005,
        auto_alpha: bool = True,
        alpha: float = 0.2,
        noise_std: float = 1.2,
        reward_normalization: bool = False,
        estimation_step: int = 1,
        deterministic_eval: bool = True,
        actor_optim: Optional[Union[Optimizer, Callable[..., Optimizer]]] = None,
        actor_optim_params: Optional[Dict[str, Any]] = None,
        critic1_optim: Optional[Union[Optimizer, Callable[..., Optimizer]]] = None,
        critic1_optim_params: Optional[Dict[str, Any]] = None,
        critic2_optim: Optional[Union[Optimizer, Callable[..., Optimizer]]] = None,
        critic2_optim_params: Optional[Dict[str, Any]] = None,
        alpha_optim: Optional[Union[Optimizer, Callable[..., Optimizer]]] = None,
        alpha_optim_params: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """[summary]

        Args:
            task_name (str): [description]
            actor (Optional[torch.nn.Module], optional): [description]. Defaults to None.
            actor_params (Optional[dict], optional): [description]. Defaults to None.
            critic1 (Optional[torch.nn.Module], optional): [description]. Defaults to None.
            critic1_params (Optional[dict], optional): [description]. Defaults to None.
            critic2 (Optional[torch.nn.Module], optional): [description]. Defaults to None.
            critic2_params (Optional[dict], optional): [description]. Defaults to None.
            gamma (float, optional): [description]. Defaults to 0.99.
            tau (float, optional): [description]. Defaults to 0.005.
            auto_alpha (bool, optional): [description]. Defaults to True.
            alpha (float, optional): [description]. Defaults to 0.2.
            noise_std (float, optional): [description]. Defaults to 1.2.
            reward_normalization (bool, optional): [description]. Defaults to False.
            estimation_step (int, optional): [description]. Defaults to 1.
            deterministic_eval (bool, optional): [description]. Defaults to True.
            actor_optim (Optional[Union[Optimizer, Callable[..., Optimizer]]], optional): [description]. Defaults to None.
            actor_optim_params (Optional[Dict[str, Any]], optional): [description]. Defaults to None.
            critic1_optim (Optional[Union[Optimizer, Callable[..., Optimizer]]], optional): [description]. Defaults to None.
            critic1_optim_params (Optional[Dict[str, Any]], optional): [description]. Defaults to None.
            critic2_optim (Optional[Union[Optimizer, Callable[..., Optimizer]]], optional): [description]. Defaults to None.
            critic2_optim_params (Optional[Dict[str, Any]], optional): [description]. Defaults to None.
            alpha_optim (Optional[Union[Optimizer, Callable[..., Optimizer]]], optional): [description]. Defaults to None.
            alpha_optim_params (Optional[Dict[str, Any]], optional): [description]. Defaults to None.















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
        actor, actor_params,
        critic1, critic1_params,
        critic2, critic2_params,
        gamma, tau,
        auto_alpha, alpha,
        noise_std,
        reward_normalization,
        estimation_step,
        deterministic_eval,
        actor_optim, actor_optim_params,
        critic1_optim, critic1_optim_params,
        critic2_optim, critic2_optim_params,
        alpha_optim, alpha_optim_params
    ):
        max_action = self.action_space.high[0]

        # actor
        self.actor_net = self.construct_rlnet(
            actor, actor_params, self.state_shape, 0
        )

        self.actor = ActorProb(
            self.actor_net, self.action_shape,
            max_action=max_action, device=self._device, unbounded=True
        ).to(self._device)
        self.actor_optim = self.construct_optim(
            actor_optim, actor_optim_params, self.actor.parameters()
        )

        # critic 1
        self.critic1_net = self.construct_rlnet(
            critic1, critic1_params, self.state_shape,
            self.action_shape, actions_as_input=True
        )

        self.critic1 = Critic(
            self.critic1_net, device=self._device).to(self._device)
        self.critic1_optim = self.construct_optim(
            critic1_optim, critic1_optim_params, self.critic1.parameters()
        )

        # critic 2
        self.critic2_net = self.construct_rlnet(
            critic2, critic2_params, self.state_shape,
            self.action_shape, actions_as_input=True
        )

        self.critic2 = Critic(
            self.critic2_net, device=self._device).to(self._device)
        self.critic2_optim = self.construct_optim(
            critic2_optim, critic2_optim_params, self.critic2.parameters()
        )

        # alpha tuning
        if auto_alpha:
            target_entropy = -np.prod(self.action_space.shape)
            log_alpha = torch.zeros(1, requires_grad=True, device=self._device)
            alpha_optim = self.construct_optim(
                alpha_optim, alpha_optim_params, [log_alpha]
            )
            alpha = (target_entropy, log_alpha, alpha_optim)

        self.policy = SACPolicy(
            self.actor, self.actor_optim,
            self.critic1, self.critic1_optim,
            self.critic2, self.critic2_optim,
            tau=tau, gamma=gamma, alpha=alpha,
            reward_normalization=reward_normalization,
            exploration_noise=OUNoise(0.0, noise_std),
            action_space=self.action_space,
            estimation_step=estimation_step,
            deterministic_eval=deterministic_eval
        )

# the simple preset

sac_simple_hyperparameters = {
    # sac
    'noise_std': 1.2,
    'gamma': 0.99,
    'tau': 0.005,
    'auto_alpha': True,
    'alpha': 0.2,
    'reward_normalization': False,
    'estimation_step': 1,
    'deterministic_eval': True,
    'actor': None,
    'actor_params': dict(model=MLP, hidden_sizes=[128, 128]),
    'actor_optim': None,
    'actor_optim_params': dict(lr=3e-4),
    'critic1': None,
    'critic1_params': dict(model=MLP, hidden_sizes=[128, 128]),
    'critic1_optim': None,
    'critic1_optim_params': dict(lr=3e-4),
    'critic2': None,
    'critic2_params': dict(model=MLP, hidden_sizes=[128, 128]),
    'critic2_optim': None,
    'critic2_optim_params': dict(lr=3e-4),
    'alpha_optim': None,
    'alpha_optim_params': dict(lr=3e-4),
    # replay buffer
    'replay_buffer': 1000000,
    'prefill_steps': None,
    # general
    'train_envs': 16,
    'test_envs': 10,
    'train_env_class': DummyVectorEnv,
    'test_env_class': DummyVectorEnv,
    'episode_per_test': None,
    'seed': None,
    'max_epoch': 10,
    'step_per_epoch': 80000,
    'step_per_collect': None,
    'update_per_step': None,
    'batch_size': 256,
    'logdir': 'log',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'train_callbacks': None,
    'test_callbacks': None,
    'train_collector': None,
    'test_collector': None,
    'exploration_noise_train': True,
    'exploration_noise_test': True,
    'task': None
}

sac_simple = AgentPreset(SACAgent, sac_simple_hyperparameters)

# the classic preset

sac_classic_hyperparameters = {
    # sac
    'noise_std': 1.2,
    'gamma': 0.99,
    'tau': 0.005,
    'auto_alpha': True,
    'alpha': 0.2,
    'reward_normalization': False,
    'estimation_step': 1,
    'deterministic_eval': True,
    'actor': None,
    'actor_params': dict(model=MLP, hidden_sizes=[128, 128]),
    'actor_optim': None,
    'actor_optim_params': dict(lr=3e-4),
    'critic1': None,
    'critic1_params': dict(model=MLP, hidden_sizes=[128, 128]),
    'critic1_optim': None,
    'critic1_optim_params': dict(lr=3e-4),
    'critic2': None,
    'critic2_params': dict(model=MLP, hidden_sizes=[128, 128]),
    'critic2_optim': None,
    'critic2_optim_params': dict(lr=3e-4),
    'alpha_optim': None,
    'alpha_optim_params': dict(lr=3e-4),
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
    'update_per_step': None,
    'batch_size': 128,
    'logdir': 'log',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'train_callbacks': None,
    'test_callbacks': None,
    'train_collector': None,
    'test_collector': None,
    'exploration_noise_train': True,
    'exploration_noise_test': True,
    'task': None
}

sac_classic = AgentPreset(SACAgent, sac_classic_hyperparameters)

# the pybullet preset

sac_pybullet_hyperparameters = {
    # sac
    'noise_std': 1.2,
    'gamma': 0.99,
    'tau': 0.005,
    'auto_alpha': True,
    'alpha': 0.2,
    'reward_normalization': False,
    'estimation_step': 1,
    'deterministic_eval': True,
    'actor': None,
    'actor_params': dict(model=MLP, hidden_sizes=[256, 256]),
    'actor_optim': None,
    'actor_optim_params': dict(lr=1e-3),
    'critic1': None,
    'critic1_params': dict(model=MLP, hidden_sizes=[256, 256]),
    'critic1_optim': None,
    'critic1_optim_params': dict(lr=1e-3),
    'critic2': None,
    'critic2_params': dict(model=MLP, hidden_sizes=[256, 256]),
    'critic2_optim': None,
    'critic2_optim_params': dict(lr=1e-3),
    'alpha_optim': None,
    'alpha_optim_params': dict(lr=3e-4),
    # replay buffer
    'replay_buffer': 1000000,
    'prefill_steps': 10000,
    # general
    'train_envs': 1,
    'test_envs': 10,
    'train_env_class': None,
    'test_env_class': None,
    'episode_per_test': None,
    'seed': None,
    'max_epoch': 200,
    'step_per_epoch': 5000,
    'step_per_collect': None,
    'update_per_step': None,
    'batch_size': 256,
    'logdir': 'log',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'train_callbacks': None,
    'test_callbacks': None,
    'train_collector': None,
    'test_collector': None,
    'exploration_noise_train': True,
    'exploration_noise_test': True,
    'task': None
}

sac_pybullet = AgentPreset(SACAgent, sac_pybullet_hyperparameters)
