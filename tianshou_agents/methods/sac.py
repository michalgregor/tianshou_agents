from ..agent import ComponentAgent
from ..components.preset import AgentPreset
from ..networks import MLP, ActorProb, Critic
from ..components import BasePolicyComponent
from tianshou.policy import SACPolicy, BasePolicy
from tianshou.exploration import BaseNoise
from typing import Any, Optional, Union, Callable, Dict
from tianshou.trainer import OffpolicyTrainer
from torch.optim import Optimizer
import numpy as np
import torch
import gym

class SACPolicyComponent(BasePolicyComponent):
    """Implementation of Soft Actor-Critic. arXiv:1812.05905.

    Args:
        agent (ComponentAgent): The agent that the component is going to
            belong into.
        device (Union[str, int, torch.device], optional): The PyTorch device
            to be used by PyTorch tensors and networks.
        seed (int, optional): The random seed to use for the component.

        observation_space (gym.spaces.Space, optional): The gym observation
            space. If None, it will be automatically derived from
            the environment in the train collector.
        action_space (gym.spaces.Space, optional): The gym action space.
            If None, it will be automatically derived from the environment
            in the train collector.
        max_epoch (Optional[int]): The maximum number of epochs for the purposes
            of constructing epsilon schedules. If not specified, it will be
            automatically derived from the trainer component. If not specified
            there either, schedule construction may fail.
        step_per_epoch (Optional[int]): The number of steps per epoch for the
            purposes of constructing epsilon schedules. If not specified, it will be
            automatically derived from the trainer component. If not specified
            there either, schedule construction may fail.
        config_arg (BasePolicy, optional): You may optionally use this argument
            to provide a pre-constructed policy to the policy component. This
            means that the policy will not be constructed automatically.
        component_class: Use this to override the default policy class for when
            the component constructs the policy object; component_class must 
            have a compatible signature to the default policy class.
        estimation_step (int): The number of steps to look ahead.
        reward_normalization (bool): normalize the reward to Normal(0, 1).
            Default to False.

        method_name: The name of the method in a format without special
            characters, so that it can be used as part of a path specification,
            etc. For SACPolicyComponent it is 'sac' by default.
        gamma (float): The discount rate; in [0, 1].

        actor (Union[torch.nn.Module, Callable[..., torch.nn.Module], Dict[str, Any]], optional):
            The torch Module to be used as the actor. Can be either
            a torch ``Module`` or ``callable(observation_space, action_space, device)``
            that returns a torch ``Module``. If None, a default RLNetwork
            is constructed.
            
            Alternatively, this can be a dictionary, where the ``type`` key
            (RLNetwork by default) is a
            ``callable(observation_space, action_space, device, **qnetwork_params)``
            and the remaining keys are ``**qnetwork_params``.
        critic1 (Union[torch.nn.Module, Callable[..., torch.nn.Module], Dict[str, Any]], optional):
            The torch Module to be used as the first critic. Can be either
            a torch ``Module`` or ``callable(observation_space, action_space, device)``
            that returns a torch ``Module``. If None, a default RLNetwork
            is constructed.
            
            Alternatively, this can be a dictionary, where the ``type`` key
            (RLNetwork by default) is a
            ``callable(observation_space, action_space, device, **qnetwork_params)``
            and the remaining keys are ``**qnetwork_params``.
        critic2 (Union[torch.nn.Module, Callable[..., torch.nn.Module], Dict[str, Any]], optional):
            The torch Module to be used as the second critic. Can be either
            a torch ``Module`` or ``callable(observation_space, action_space, device)``
            that returns a torch ``Module``. If None, a default RLNetwork
            is constructed.
            
            Alternatively, this can be a dictionary, where the ``type`` key
            (RLNetwork by default) is a
            ``callable(observation_space, action_space, device, **qnetwork_params)``
            and the remaining keys are ``**qnetwork_params``.
        tau (float, optional): Param for the soft update of the target network.
        alpha (float, optional): [description]. The entropy regularization
            coefficient. Note: if ``auto_alpha`` is ``True``, this is
            tuned automatically during training.
        auto_alpha (bool, optional): Specifies whether alpha should be
            tuned automatically during training.
        target_entropy (float, optional): Specifies the target entropy;
            if not None, this automatically toggles on auto_alpha=True;
            if None, and auto_alpha == True, it is set to
            -gym.spaces.flatdim(self.action_space).
        exploration_noise (BaseNoise, optional):
            Noise added to actions for exploration. This is useful when
            solving hard-exploration problems. By default this is None,
            which means no noise is going to be used. If you use noise,
            you need to ensure that its scale w.r.t. the scale of the
            actions is appropriate.
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

        **kwargs: Any additional keyword arguments are passed to the
            ``__init__`` method of the policy class.
    """

    def __init__(
        self,
        # component args
        agent: ComponentAgent,
        device: Union[str, int, torch.device] = "cpu",
        seed: int = None,
        # context args
        max_epoch: Optional[int] = None,
        step_per_epoch: Optional[int] = None,
        config_arg: Optional[BasePolicy] = None,
        component_class: Any = None,
        estimation_step: int = 1,
        reward_normalization: bool = False,
        # method args
        method_name: str = "sac",
        gamma: float = 0.99,
        actor: Optional[Union[torch.nn.Module, Callable[..., torch.nn.Module], Dict[str, Any]]] = None,
        critic1: Optional[Union[torch.nn.Module, Callable[..., torch.nn.Module], Dict[str, Any]]] = None,
        critic2: Optional[Union[torch.nn.Module, Callable[..., torch.nn.Module], Dict[str, Any]]] = None,
        tau: float =  0.005,
        alpha: float = 0.2,
        auto_alpha: bool = True,        
        target_entropy: Optional[float] = None,
        exploration_noise: Optional[BaseNoise] = None,       
        deterministic_eval: bool = True,
        actor_optim: Optional[Union[Optimizer, Callable[..., Optimizer], Dict[str, Any]]] = None,
        critic1_optim: Optional[Union[Optimizer, Callable[..., Optimizer], Dict[str, Any]]] = None,
        critic2_optim: Optional[Union[Optimizer, Callable[..., Optimizer], Dict[str, Any]]] = None,
        alpha_optim: Optional[Union[Optimizer, Callable[..., Optimizer], Dict[str, Any]]] = None,
        observation_space: Optional[gym.spaces.Space] = None,
        action_space: Optional[gym.spaces.Space] = None,
        **policy_kwargs
    ):
        super().__init__(method_name=method_name)

        # the sate dict
        self._state_objs.extend([
            'policy',
            'policy.actor_optim',
            'policy.critic1_optim',
            'policy.critic2_optim',
            'alpha_optim'
        ])

        # the algo
        if isinstance(config_arg, BasePolicy):
            self.policy = config_arg
        else:
            # setup the default policy class
            if component_class is None:
                component_class = SACPolicy

            observation_space = agent.get_observation_space(observation_space)
            action_space = agent.get_action_space(action_space)
            max_action = np.max(action_space.high)

            # actor
            self.actor_net = self.construct_rlnet(
                module=actor,
                observation_space=observation_space,
                action_space=None,
                device=device
            )

            self.actor = ActorProb(
                self.actor_net, gym.spaces.flatten_space(action_space).shape,
                max_action=max_action, device=device, unbounded=True
            ).to(device)

            self.actor_optim = self.construct_optim(
                actor_optim, self.actor.parameters()
            )

            # critic 1
            self.critic1_net = self.construct_rlnet(
                module=critic1,
                observation_space=observation_space,
                action_space=action_space,
                actions_as_input=True,
                device=device
            )

            self.critic1 = Critic(
                self.critic1_net, device=device).to(device)

            self.critic1_optim = self.construct_optim(
                critic1_optim, self.critic1.parameters()
            )

            # critic 2
            self.critic2_net = self.construct_rlnet(
                module=critic2,
                observation_space=observation_space,
                action_space=action_space,
                actions_as_input=True,
                device=device
            )

            self.critic2 = Critic(
                self.critic2_net, device=device).to(device)

            self.critic2_optim = self.construct_optim(
                critic2_optim, self.critic2.parameters()
            )

            # alpha tuning
            self.alpha_optim = None

            if auto_alpha or not target_entropy is None:
                if target_entropy is None:
                    target_entropy = -gym.spaces.flatdim(action_space)

                log_alpha = torch.zeros(1, requires_grad=True, device=device)
                self.alpha_optim = self.construct_optim(
                    alpha_optim, [log_alpha]
                )
                alpha = (target_entropy, log_alpha, self.alpha_optim)

            self.policy = component_class(
                self.actor, self.actor_optim,
                self.critic1, self.critic1_optim,
                self.critic2, self.critic2_optim,
                tau=tau, gamma=gamma, alpha=alpha,
                reward_normalization=reward_normalization,
                exploration_noise=exploration_noise,
                action_space=action_space,
                estimation_step=estimation_step,
                deterministic_eval=deterministic_eval,
                **policy_kwargs
            )

# base config

sac_base_config = {
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
    'policy': SACPolicyComponent,
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
    # trainer
    'trainer': {},
    'trainer_class': OffpolicyTrainer,
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
    'logger': 'log'
}

sac_default = AgentPreset(sac_base_config)

# classic

sac_classic_hyperparameters = sac_default.derive_conf({
    'train_envs': 16,
    'test_envs': 100
})

sac_classic = AgentPreset(sac_classic_hyperparameters)

# simple

sac_simple_hyperparameters = sac_default.derive_conf({
    'test_envs': 5
})

sac_simple = AgentPreset(sac_simple_hyperparameters)

# pybullet

sac_pybullet_hyperparameters = sac_default.derive_conf({
    # sac
    'actor': dict(model=MLP, hidden_sizes=[256, 256]),
    'actor_optim': dict(lr=1e-3),
    'critic1': dict(model=MLP, hidden_sizes=[256, 256]),
    'critic1_optim': dict(lr=1e-3),
    'critic2': dict(model=MLP, hidden_sizes=[256, 256]),
    'critic2_optim': dict(lr=1e-3),
    # replay buffer
    'prefill_steps': 10000,
    # general
    'test_envs': 10,
    'max_epoch': 200,
    'step_per_epoch': 5000,
    'batch_size': 256
})

sac_pybullet = AgentPreset(sac_pybullet_hyperparameters)
