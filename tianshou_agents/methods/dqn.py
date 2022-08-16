from ..agent import ComponentAgent
from ..components.preset import AgentPreset
from ..components import BasePolicyComponent
from ..schedules import Schedule, ConstSchedule, ExponentialSchedule
from ..callbacks import ScheduleCallback
from ..networks import MLP
from tianshou.trainer import OffpolicyTrainer
from tianshou.policy import DQNPolicy, BasePolicy
from typing import Any, Optional, Union, Callable, Dict, Sequence
from torch.optim import Optimizer
from numbers import Number
import torch
import gym

class DQNPolicyComponent(BasePolicyComponent):
    """Implementation of Deep Q Network. arXiv:1312.5602.

    Implementation of Double Q-Learning. arXiv:1509.06461.

    Implementation of Dueling DQN. arXiv:1511.06581 (the dueling DQN is
    implemented in the network side, not here).

    Args:
        agent (ComponentAgent): The agent that the component is going to
            belong into.
        device (Union[str, int, torch.device], optional): The PyTorch device
            to be used by PyTorch tensors and networks.
        seed (int, optional): The random seed to use for the component.

        observation_space (gym.spaces.Space, optional): The observation
            space. If None, it will be automatically copied from the
            environment in the train collector.
        action_space (gym.spaces.Space, optional): The action space. If None,
            it will be automatically copied from the environment in the train
            collector.
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
            etc. For DQNPolicyComponent it is 'dqn' by default.
        discount_factor (float): The discount rate; in [0, 1].
        qnetwork (Union[torch.nn.Module, Callable[..., torch.nn.Module],
            Dict[str, Any]], optional): The torch Module to be used as the
            Q-Network. Can be either a torch Module or
            callable(state_space, action_space, device) that returns a torch
            Module. If None, a default RLNetwork is constructed.
            
            Alternatively, this can be a dictionary, where the type key
            (RLNetwork by default) is a
            callable(state_space, action_space, device, **qnetwork_params)
            and the remaining keys are **qnetwork_params.
        optim (Union[Optimizer, Callable[..., Optimizer],
            Dict[str, Any]], optional): The optimizer to use for training the
            Q-Network. This can either be an Optimizer instance or
            callable(parameters). Defaults to None, which means the Adam
            optimizer is going to be constructed.

            Alternatively, this can be a dictionary, where the type key
            (Adam) is a callable(parameters, **kwargs) and the 
            remaining keys are **kwargs.
        target_update_freq (int): The target network update frequency
            (0 if you do not use the target network).
        is_double (bool): use double dqn. Default to True.
        eps_test (Union[float, Schedule, Callable[[int, int], Schedule]]):
            The test-time epsilon (exploration rate); can be a float,
            a schedule or a ``callable(max_epoch, step_per_epoch)``
            that constructs a schedule.
        eps_train (Union[float, Schedule, Callable[[int, int], Schedule]]):
            The train-time epsilon (exploration rate); can be a float,
            a schedule or a ``callable(max_epoch, step_per_epoch)``
            that constructs a schedule.
        
        **kwargs: Any additional keyword arguments are passed to the
            ``__init__`` method of the policy class.
    """

    def __init__(
        self,
        # component args
        agent: ComponentAgent,
        device: Optional[Union[str, int, torch.device]] = None,
        seed: Optional[int] = None,
        # context args
        max_epoch: Optional[int] = None,
        step_per_epoch: Optional[int] = None,
        config_arg: Optional[BasePolicy] = None,
        component_class: Any = None,
        estimation_step: int = 1,
        reward_normalization: bool = False,
        # method args
        method_name: str = "dqn",
        discount_factor: float = 0.99,
        qnetwork: Optional[Union[torch.nn.Module, Callable[..., torch.nn.Module], Dict[str, Any]]] = None,
        optim: Optional[Union[Optimizer, Callable[..., Optimizer], Dict[str, Any]]] = {},
        target_update_freq: int = 0,
        is_double: bool = True,
        eps_train: Union[float, Schedule, Callable[[int, int], Schedule]] = 
            lambda max_epoch, step_per_epoch: ExponentialSchedule(
                max_epoch*step_per_epoch*0.5, 0.73, 0.1
            ),
        eps_test: Union[float, Schedule, Callable[[int, int], Schedule]] = 0.01,
        observation_space: Optional[gym.spaces.Space] = None,
        action_space: Optional[gym.spaces.Space] = None,
        **policy_kwargs
    ):
        super().__init__(method_name=method_name)

        # the state dict
        self._state_objs.extend([
            'policy',
            'policy.optim',
        ])

        # the algo
        if isinstance(config_arg, BasePolicy):
            self.policy = config_arg
        else:
            # setup the default policy class
            if component_class is None:
                component_class = DQNPolicy

            # the network
            qnetwork = self.construct_rlnet(
                module=qnetwork,
                observation_space=agent.get_observation_space(observation_space),
                action_space=agent.get_action_space(action_space),
                device=device
            )

            optim = self.construct_optim(
                optim, qnetwork.parameters()
            )

            self.policy = component_class(
                model=qnetwork,
                optim=optim,
                discount_factor=discount_factor,
                estimation_step=estimation_step,
                target_update_freq=target_update_freq,
                reward_normalization=reward_normalization,
                is_double=is_double,
                **policy_kwargs
            )

        # eps schedules
        if max_epoch is None:
            max_epoch = agent.component_trainer.max_epoch

        if step_per_epoch is None:
            step_per_epoch = agent.component_trainer.step_per_epoch

        if isinstance(eps_train, Schedule):
            eps_train = eps_train
        elif isinstance(eps_train, Number):
            eps_train = ConstSchedule(eps_train)
        else:
            eps_train = eps_train(max_epoch, step_per_epoch)

        agent.train_callbacks.append(ScheduleCallback(self.policy.set_eps, eps_train))

        if isinstance(eps_test, Schedule):
            eps_test = eps_test
        elif isinstance(eps_test, Number):
            eps_test = ConstSchedule(eps_test)
        else:
            eps_test = eps_test(max_epoch, step_per_epoch)

        agent.test_callbacks.append(ScheduleCallback(self.policy.set_eps, eps_test))

# base config

dqn_base_config = {
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
    'policy': DQNPolicyComponent,
    'is_double': True,
    'eps_test': 0.01,
    'eps_train': lambda max_epoch, step_per_epoch: ExponentialSchedule(
            max_epoch*step_per_epoch*0.5, 0.73, 0.1),
    'discount_factor': 0.99,
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

dqn_default = AgentPreset(dqn_base_config)

# classic
dqn_classic_hyperparameters = dqn_default.derive_conf({
    'train_envs': 1,
    'test_envs': 100
})

dqn_classic = AgentPreset(dqn_classic_hyperparameters)

# simple

dqn_simple_hyperparameters = dqn_default.derive_conf({
    'train_envs': 1,
    'test_envs': 5
})

dqn_simple_hyperparameters["policy"]["qnetwork"]["dueling_param"] = None

dqn_simple = AgentPreset(dqn_simple_hyperparameters)
