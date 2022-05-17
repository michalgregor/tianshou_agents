from typing import Optional, Union, Callable, Dict, Any, Tuple
from ..components.trainer import TrainerComponent
from ..components.replay_buffer import BaseReplayBufferComponent, ReplayBufferComponent
from ..components.collector import CollectorComponent
from ..components.policy import BasePolicyComponent
from ..components.logger import LoggerComponent
from ..components.passive_interface import PassiveInterface
from tianshou.policy import BasePolicy
from ..utils import ConfigBuilder
from tianshou.utils.logger.base import BaseLogger
from tianshou.data import ReplayBuffer, Collector
import torch
import gym

class BaseConfigRouter:
    """
    A convenience class that enables specification of some component parameters
    at the agent-level and then takes care of routing them to their respective
    components, producing a regular component-wise config.

    ComponentAgent can even be customized by passing a custom
    ConfigRouter to it – that way it can theoretically support different sets
    of agent-level args for different agent presets, use cases, etc.

    The config router class has several ConfigBuilder attributes that are
    used to construct the config for each component: replay_buffer_builder,
    collector_builder, policy_builder, trainer_builder, logger_builder.

    The BaseConfigRouter actually does not do any routing and does not accept
    any agent-level arguments. It only transforms the standard component
    configs into dictionary format and returns the resulting agent 
    configuration as a dict.

    For description of the basic component agent args, see the docstring of
    the ComponentAgent class.
    """

    replay_buffer_builder = ConfigBuilder(
        obj_type=BaseReplayBufferComponent,
        default_obj_constructor=ReplayBufferComponent
    )

    collector_builder = ConfigBuilder(
        obj_type=CollectorComponent,
        default_obj_constructor=CollectorComponent
    )

    policy_builder = ConfigBuilder(
        obj_type=BasePolicyComponent
    )

    logger_builder = ConfigBuilder(
        obj_type=LoggerComponent,
        default_obj_constructor=LoggerComponent
    )

    trainer_builder = ConfigBuilder(
        obj_type=TrainerComponent,
        default_obj_constructor=TrainerComponent
    )

    passive_interface_builder = ConfigBuilder(
        obj_type=PassiveInterface,
        default_obj_constructor=PassiveInterface
    )

    def __call__(self,
        # components
        replay_buffer: Optional[Union[
            int,
            ReplayBuffer,
            BaseReplayBufferComponent,
            Callable[..., BaseReplayBufferComponent],
            Dict[str, Any]
        ]] = None,
        train_collector: Union[
            Collector,
            CollectorComponent,
            Callable[..., CollectorComponent],
            Dict[str, Any]
        ] = None,
        test_collector: Union[
            Collector,
            CollectorComponent,
            Callable[..., CollectorComponent],
            Dict[str, Any]
        ] = None,
        policy: Union[
            BasePolicy,
            BasePolicyComponent,
            Callable[..., BasePolicyComponent],
            Dict[str, Any]
        ] = None,
        logger: Union[
            str,
            BaseLogger,
            LoggerComponent,
            Callable[..., LoggerComponent],
            Dict[str, Any]
        ] = {},
        trainer: Union[
            TrainerComponent,
            Callable[..., TrainerComponent],
            Dict[str, Any]
        ] = None,
        passive_interface: Union[
            PassiveInterface,
            Callable[..., PassiveInterface],
            Dict[str, Any]
        ] = None,
        # agent args
        device: Optional[Union[str, int, torch.device]] = None,
        seed: Optional[int] = None,
        extract_obs_shape: Callable[[gym.spaces.Space], Tuple[int, ...]] = None,
        extract_act_shape: Callable[[gym.spaces.Space], Tuple[int, ...]] = None,
    ):
        return dict(
            replay_buffer=self.replay_buffer_builder.to_dict_config(replay_buffer),
            train_collector=self.train_collector_builder.to_dict_config(train_collector),
            test_collector=self.test_collector_builder.to_dict_config(test_collector),
            policy=self.policy_builder.to_dict_config(policy),
            logger=self.logger_builder.to_dict_config(logger),
            trainer=self.trainer_builder.to_dict_config(trainer),
            passive_interface=self.passive_interface_builder.to_dict_config(passive_interface),
            device=device,
            seed=seed,
            extract_obs_shape=extract_obs_shape,
            extract_act_shape=extract_act_shape
        )

class DefaultConfigRouter(BaseConfigRouter):
    """
    A convenience class that enables specification of some component parameters
    at the agent-level and then takes care of routing them to their respective
    components, producing a regular component-wise config.

    ComponentAgent can even be customized by passing a custom
    ConfigRouter to it – that way it can theoretically support different sets
    of agent-level args for different agent presets, use cases, etc.

    The config router class has several ConfigBuilder attributes that are
    used to construct the config for each component: replay_buffer_builder,
    collector_builder, policy_builder, trainer_builder, logger_builder.

    For description of the basic component agent args, see the docstring of
    the ComponentAgent class.

    Agent-level args for the collectors:

        train_envs (Union[
            int,
            List[
                Union[
                    gym.Env,
                    Callable[[], gym.Env]
                ]
            ],
            BaseVectorEnv
        ], optional):
            A spec used to construct training environments. One of:
                * None;
                * The number of environments to construct;
                * A list of Gym environments / callables that construct
                    Gym environments;
                * A BaseVectorEnv instance.
            When the environments are constructed automatically, it is using
            the class specified through the train_env_class argument.

        test_envs (Union[
            int,
            List[
                Union[
                    gym.Env,
                    Callable[[], gym.Env]
                ]
            ],
            BaseVectorEnv
        ], optional):
            A spec used to construct testing environments. One of:
                * None;
                * The number of environments to construct;
                * A list of Gym environments / callables that construct
                    Gym environments;
                * A BaseVectorEnv instance.
            When the environments are constructed automatically, it is using
            the class specified through the train_env_class argument.

        train_env_class (Type[BaseVectorEnv], optional):
            The vector environment used to represent the collection of training
            environments. See also ``train_envs``.

        test_env_class (Type[BaseVectorEnv], optional):
            The vector environment used to represent the collection of testing
            environments. See also ``test_envs``.
    
        exploration_noise_train (bool, optional): Determines whether, during
            training, the action needs to be modified with the corresponding
            policy's exploration noise. If so, ``policy.exploration_noise(act, batch)``
            will be called automatically to add the exploration noise into the
            action. This is only used unless a pre-constructed collector is
            supplied.

        exploration_noise_test (bool, optional): Determines whether, during
            testing, the action needs to be modified with the corresponding
            policy's exploration noise. If so, ``policy.exploration_noise(act, batch)``
            will be called automatically to add the exploration noise into the
            action. This is only used unless a pre-constructed collector is
            supplied.
       
        task_name (str): The name of the ``gym`` environment; by default,
            environments are constructed using ``gym.make``. To override
            this behaviour, supply a ``task`` argument: a callable that
            constructs your ``gym`` environment.

        task (Callable[[], gym.Env], optional): A callable used to
            construct training and testing environments. Defaults to None
            in which case evironments are constructed using
            gym.make(task_name).

        train_task (Callable[[], gym.Env], optional): A callable used to
            construct training environments. Defaults to None
            in which case evironments are constructed using
            gym.make(task_name). If both task and train_task are specified,
            train_task takes precedence.

        test_task (Callable[[], gym.Env], optional): A callable used to
            construct testing environments. Defaults to None in which
            case testing environments are constructed using
            gym.make(task_name). If both task and test_task are specified,
            test_task takes precedence.

    Agent-level args for the trainer:

        trainer_class: The trainer class to use. This is routed to the
            component_class argument of the trainer component.

        max_epoch (int, optional): The maximum number of epochs for training.
            The training process might be finished before reaching
            ``max_epoch`` if the stop criterion returns ``True``; this
            behaviour can be overriden using the ``stop_criterion`` argument.

        step_per_epoch (int, optional): The number of transitions collected
            per epoch.

        step_per_collect (int, optional): The number of transitions a collector
            would collect before the network update, i.e., trainer will
            collect ``step_per_collect`` transitions and do some policy
            network update repeatly in each epoch. Defaults to ``None``, 
            which means that ``step_per_collect`` is the same as the
            number of training environments.

        update_per_collect (float, optional): The number of times the policy
            network will be updated for each collect (collection of
            experience from the environments). If update_per_step is
            specified, it overrides this argument. Defaults to 1.

        update_per_step (float, optional): The number of times the policy
            network would be updated per transition after (step_per_collect)
            transitions are collected, e.g., if update_per_step set to 0.3,
            and step_per_collect is 256, policy will be updated
            round(256 * 0.3 = 76.8) = 77 times after 256 transitions are
            collected by the collector. If None (default), it is calculated
            automatically from update_per_collect.
        
        episode_per_test (int, optional): The number of episodes for one
            policy evaluation.

        verbose (bool): whether to print information when training.
            Defaults to True.

        show_progress (bool): whether to display a progress bar when training.
            Defaults to True.

        reward_metric: a function with signature
            ``f(rewards: np.ndarray with shape (num_episode, agent_num)) -> np.ndarray
            with shape (num_episode,)``, used in multi-agent RL. We need to 
            return a single scalar for each episode's result to monitor
            training in the multi-agent RL setting. This function specifies
            what is the desired metric, e.g., the reward of agent 1 or the
            average reward over all agents.

        batch_size (int): The batch size of sample data, which is going to
        feed in the policy network.

        repeat_per_collect (int, optional): The number of repeat time for
            policy learning, for example, set it to 2 means the policy needs
            to learn each given batch data twice.
        
        update_per_epoch  The number of policy network updates,
            so-called gradient steps, per epoch.

        episode_per_collect (int, optional): The number of episodes the
            collector would collect before the network update, i.e., trainer
            will collect "episode_per_collect" episodes and do some policy
            network update repeatedly in each epoch.

        test_in_train (bool): Whether to test in the training phase.

        prefill_steps (int): The number of steps to prefill the replay
            buffer. The prefilling is done when the first trainer is constructed
            in the trainer component. Prefilling is only done if a train
            collector is available. It is not done if the agent is run in
            passive mode – the user provides the data then and is responsible
            for prefilling the buffer when necessary.

        resume_from_log (bool): Whether trainer's counters should resume from the
            global counters in the logger. If False, the trainer will maintain
            its own counters. This will, however, have no effect on schedulers
            and other callbacks, which are still going to get info from the
            global counters.

            This should be set to False if you want to interleave several
            different trainers – in that case each should iterate from 0 to its
            own local max_epoch – since the different trainers will not sync
            their epoch counters they would not respect a global max_epoch
            setting.

            This can also be specified as a keyword argument when creating
            a trainer using make_trainer.

        reward_threshold (float, optional): The reward threshold to use for
            the stop criterion (if any). Note that richer stopping criteria can
            be specified using the stop_criterion argument.

        train_callbacks (Union[
            List[CallbackType],
            Callable[..., Union[CallbackType, List[CallbackType]]],
            Dict[str, Any]
        ], optional): A list of callbacks invoked at the beginning of each
            training step. The signature of the callbacks is
            ``f(epoch: int, env_step: int, gradient_step: int, agent: ComponentAgent) -> None``.

            Optionally, save callbacks can be constructed using
            a ConfigBuilder spec.

        test_callbacks (Union[
            List[CallbackType],
            Callable[..., Union[CallbackType, List[CallbackType]]],
            Dict[str, Any]
        ], optional): A list of callbacks invoked at the beginning of each
            testing step. The signature of the callbacks is
            ``f(epoch: int, env_step: int, gradient_step: int, agent: ComponentAgent) -> None``.

            Optionally, save callbacks can be constructed using
            a ConfigBuilder spec.

        save_best_callbacks (Union[
            str,
            List[CallbackType],
            Callable[..., Union[CallbackType, List[CallbackType]]],
            Dict[str, Any]
        ], optional): A list of callbacks invoked every time the undiscounted
            average mean reward in evaluation phase gets better.

            The signature of the callbacks is
            ``f(epoch: int, env_step: int, gradient_step: int, agent: ComponentAgent) -> None``.

            Optionally, a callback can be constructed automatically using
            a ConfigBuilder spec.
            
            In that case a string other than 'auto' is interpreted as
            specifying the log_path argument.

            If log_path is not specified in this way or through the dict
            argument, the trainer will try to use the log_path from the logger, 
            if available.

        save_checkpoint_callbacks (Union[
            str,
            List[CallbackType],
            Callable[..., Union[CallbackType, List[CallbackType]]],
            Dict[str, Any]
        ], optional): A list of callbacks invoked after every step of training.

            The signature of the callbacks is
            ``f(epoch: int, env_step: int, gradient_step: int, agent: ComponentAgent) -> None``.

            Optionally, a save callback can be constructed automatically using
            a ConfigBuilder spec. Unless otherwise specified, the
            interval is going to default to max(int(self.max_epoch / 10), 1)
            if max_epoch is specified and to 1 otherwise.

            If a callback is being constructed automatically, a string other
            than 'auto' is interpreted as specifying the log_path argument.

            If log_path is not specified in this way or through the dict
            argument, the trainer will try to use the log_path from the logger, 
            if available.
                
        stop_criterion (
            Union[
                bool,
                str,
                Callable[[float, Union[float, None], 'ComponentAgent'], float]
            ], optional
        ):
            The criterion used to stop training before ``max_epoch`` has
            been reached:
                * If set to ``True``, training stops once the
                    mean test reward from the previous collect reaches the
                    environment's reward threshold;
                * If set to ``False``, the stop criterion is disabled;
                * If a float, training stops once the mean test reward
                    from the previous collect reaches ``stop_criterion``.
                * If set to ``callable(mean_rewards, reward_threshold, agent)``,
                    the callable is used to determine whether training should
                    be stopped or not; mean_rewards is the mean test reward
                    from the previous collect.

    Agent-level args for the passive interface:
        passive_collector: The PassiveCollector to use for collecting data
            in passive mode.

        passive_trainer: The trainer to use for training in passive mode.
            This is an instance of BaseTrainer that is going to be wrapped
            in a StepWiseTrainer internally.

    Agent-level args for the policy:

        **policy_kwargs (Any): Any other keyword arguments are passed to
            the policy component.
    """

    # equivalent keys
    # kwargs key: component agent key
    key_equivalence = {
        'replay_buffer': 'replay_buffer',
        'train_collector': 'train_collector',
        'test_collector': 'test_collector',
        'policy': 'policy',
        'logger': 'logger',
        'trainer': 'trainer',
        'passive_interface': 'passive_interface',
        'device': 'device',
        'seed': 'seed',
        'extract_obs_shape': 'extract_obs_shape',
        'extract_act_shape': 'extract_act_shape',
    }

    # subkey routing
    # component agent key: {kwargs key: component agent subkey}
    subkey_routing = {
        'replay_buffer': {
            'num_envs': 'num_envs',
        },
        'train_collector': {
            'train_envs': 'env',
            'train_env_class': 'env_class',
            'exploration_noise_train': 'exploration_noise',
            'task_name': 'task_name',
            'task': 'task',
            'train_task': 'task',
        },
        'test_collector': {
            'test_envs': 'env',
            'test_env_class': 'env_class',
            'exploration_noise_test': 'exploration_noise',
            'task_name': 'task_name',
            'task': 'task',
            'test_task': 'task',
        },
        'trainer': {
            'trainer_class': 'component_class',
            'max_epoch': 'max_epoch',
            'step_per_epoch': 'step_per_epoch',
            'step_per_collect': 'step_per_collect',
            'update_per_collect': 'update_per_collect',
            'update_per_step': 'update_per_step',
            'episode_per_test': 'episode_per_test',
            'verbose': 'verbose',
            'show_progress': 'show_progress',
            'reward_metric': 'reward_metric',
            'batch_size': 'batch_size',
            'repeat_per_collect': 'repeat_per_collect',
            'update_per_epoch': 'update_per_epoch',
            'episode_per_collect': 'episode_per_collect',
            'test_in_train': 'test_in_train',
            'prefill_steps': 'prefill_steps',
            'resume_from_log': 'resume_from_log',
            'reward_threshold': 'reward_threshold',
            'train_callbacks': 'train_callbacks',
            'test_callbacks': 'test_callbacks',
            'save_best_callbacks': 'save_best_callbacks',
            'save_checkpoint_callbacks': 'save_checkpoint_callbacks',
            'stop_criterion': 'stop_criterion',
        },
        'passive_interface': {
            'passive_collector': 'config_arg',
            'passive_trainer': 'trainer',
            'num_envs': 'num_envs',
        },
        'logger': {
            'task_name': 'task_name',
        }
    }

    # the component agent key to which unused kwargs should be routed
    fallthrough_key = 'policy'
    
    def __call__(self, **kwargs):
        # component agent key: builder
        builders = {
            'replay_buffer': self.replay_buffer_builder,
            'train_collector': self.collector_builder,
            'test_collector': self.collector_builder,
            'policy': self.policy_builder,
            'logger': self.logger_builder,
            'trainer': self.trainer_builder,
            'passive_interface': self.passive_interface_builder,
        }

        # create an empty config
        config = {}
        
        # get the kwargs keys as a set
        kwargs_keys = {*kwargs}

        # used keys
        used_keys = set()

        # check which component agent keys are in kwargs,
        # use them to init the config dict
        equiv_component_keys = {*self.key_equivalence} & kwargs_keys
        used_keys |= equiv_component_keys

        # construct a base conf using equivalent keys
        for key in equiv_component_keys:
            equiv_key = self.key_equivalence[key]
            config[equiv_key] = kwargs[key]
            builder = builders.get(equiv_key, None)
            if builder is not None:
                config[equiv_key] = builder.to_dict_config(config[equiv_key])

        # route the subkeys
        for component_key, subkey_dict in self.subkey_routing.items():
            subkey_intersect = {*subkey_dict} & kwargs_keys
            used_keys |= subkey_intersect

            if not len(subkey_intersect): continue            
            component_dict = config[component_key]
            
            # components marked as None don't receive arguments:
            if component_dict is None: continue
            
            for subkey in subkey_intersect:
                component_dict[subkey_dict[subkey]] = kwargs[subkey]

        # route the unused kwargs
        unused_keys = kwargs_keys - used_keys
        fallthrough_subdict = config.get(self.fallthrough_key, {})
        config[self.fallthrough_key] = fallthrough_subdict
        
        for key in unused_keys:
            fallthrough_subdict[key] = kwargs[key]

        return config
