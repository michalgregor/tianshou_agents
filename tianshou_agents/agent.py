from .utils import StateDictObject, construct_config_object
from .components.replay_buffer import BaseReplayBufferComponent, ReplayBufferComponent
from .components.collector import CollectorComponent, PassiveCollector, Batch, DummyCollector
from .components.policy import BasePolicyComponent
from .components.logger import LoggerComponent
from .components.trainer import TrainerComponent, CallbackType, StepWiseTrainer
from tianshou.data import ReplayBuffer, Collector
from tianshou.policy import BasePolicy
from tianshou.trainer import BaseTrainer
from tianshou.env import BaseVectorEnv
from tianshou.utils.logger.base import BaseLogger
from .components.env import extract_shape

from typing import Optional, Union, Callable, Dict, Any, List, Type, Tuple
from functools import partial
import numpy as np
import warnings
import torch
import gym
import abc

class BaseAgent(StateDictObject):
    @abc.abstractmethod
    def reset_progress_counters(self):
        """Resets progress counters (epoch, env_step, ...)"""

    @abc.abstractmethod
    def make_trainer(self, **kwargs) -> BaseTrainer:
        """Creates a trainer. The keyword arguments (if any) are used to
        override the arguments originally passed to the trainer.
        Most notably, perhaps, you can override the:
            * ``max_epoch``;
            * ``batch_size``;
            * ``step_per_epoch``;
            * ``step_per_collect``;
            * ...

        Returns:
            A trainer.
        """
    
    @abc.abstractmethod
    def train(self, **kwargs) -> Dict[str, Union[float, str]]:
        """Runs training. The keyword arguments (if any) are used to
        override the arguments originally passed to the trainer.
        Most notably, perhaps, you can override the:
            * ``max_epoch``;
            * ``batch_size``;
            * ``step_per_epoch``;
            * ``step_per_collect``;
            * ...

        Returns:
            dict: See :func:`~tianshou.trainer.gather_info`.
        """
        
    @abc.abstractmethod
    def test(self,
        render: Optional[float] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Runs testing and returns the results from the test collector.

        Args:
            render (float, optional): The sleep time between rendering
                consecutive frames. Default to None (no rendering).
            seed (int, optional): The numeric seed used to seed the random
                number generators (``np.random``, ``torch.manual_seed``,
                ``train_envs.seed``, ``test_envs.seed``). Defaults to ``None``
                in which case no seeding is done.

        Returns:
            Dict[str, Any]: The results from the test collector.
        """

class BasePassiveAgent(StateDictObject):
    @abc.abstractmethod
    def init_passive_training(self):
        pass

    @abc.abstractmethod
    def finish_passive_training(self):
        pass

    @abc.abstractmethod
    def act(
        self,
        obs: Union[Batch, np.ndarray, torch.Tensor],
        done: Optional[Union[Batch, np.ndarray, torch.Tensor]],
        state: Optional[Batch],
        no_grad: bool = True,
        return_info: bool = False,
    ) -> Tuple[Batch, Batch]:
        pass

    @abc.abstractmethod
    def observe_transition(
        self,
        data: Batch,
        env_ids: Optional[Union[np.ndarray, List[int]]] = None
    ) -> None:
        pass

class ComponentAgent(BaseAgent, BasePassiveAgent):
    """
    An agent class that builds the agent from several components:
        - replay buffer: stores the collected experience;
        - train collector: collects the experience from the environment
            for training (this is optional – it is also possible to populate
            the replay buffer directly);
        - test collector: collects the experience from the environment (this
            is optional – if a testing collector is not provided, testing
            is skipped);
        - policy: the component that constructs the policy to be trained;
        - logger: the component that logs the training progress;
        - trainer: the component that trains the policy.

    Args:
        component_replay_buffer (Union[
            int,
            ReplayBuffer,
            BaseReplayBufferComponent,
            Callable[..., BaseReplayBufferComponent],
            Dict[str, Any]
        ]): The replay buffer component in a spec for construct_config_object.

        component_train_collector (Union[
            Collector,
            CollectorComponent,
            Callable[..., CollectorComponent],
            Dict[str, Any]
        ]) The train collector component in a spec for construct_config_object.

        component_test_collector (Union[
            Collector,
            CollectorComponent,
            Callable[..., CollectorComponent],
            Dict[str, Any]
        ]) The test collector component in a spec for construct_config_object.

        component_policy (Union[
            BasePolicy,
            BasePolicyComponent,
            Callable[..., BasePolicyComponent],
            Dict[str, Any]
        ]) The policy component in a spec for construct_config_object.

        component_logger (Union[
            str,
            BaseLogger,
            LoggerComponent,
            Callable[..., LoggerComponent],
            Dict[str, Any]
        ]) The logger component in a spec for construct_config_object.

        component_trainer (Union[
            TrainerComponent,
            Callable[..., TrainerComponent],
            Dict[str, Any]
        ]) The trainer component in a spec for construct_config_object.

        device (Union[str, int, torch.device], optional): The PyTorch device
            to be used by PyTorch tensors and networks.
        seed (int, optional): The numeric seed used to seed the random
                number generators (``np.random``, ``torch.manual_seed``,
                ``train_envs.seed``, ``test_envs.seed``). Defaults to ``None``
                in which case no seeding is done.

        replay_buffer_kwargs (dict, optional): Additional keyword arguments to
            be passed to construct_config_object when constructing the replay
            buffer component.
        train_collector_kwargs (dict, optional): Additional keyword arguments to
            be passed to construct_config_object when constructing the train
            collector component.
        test_collector_kwargs (dict, optional): Additional keyword arguments to
            be passed to construct_config_object when constructing the test
            collector component.
        policy_kwargs (dict, optional): Additional keyword arguments to be
            passed to construct_config_object when constructing the policy
            component.
        logger_kwargs (dict, optional): Additional keyword arguments to be
            passed to construct_config_object when constructing the logger
            component.
        trainer_kwargs (dict, optional): Additional keyword arguments to be
            passed to construct_config_object when constructing the trainer
            component.
    """
    def __init__(self,
        component_replay_buffer: Union[
            int,
            ReplayBuffer,
            BaseReplayBufferComponent,
            Callable[..., BaseReplayBufferComponent],
            Dict[str, Any]
        ],
        component_train_collector: Union[
            Collector,
            CollectorComponent,
            Callable[..., CollectorComponent],
            Dict[str, Any]
        ],
        component_test_collector: Union[
            Collector,
            CollectorComponent,
            Callable[..., CollectorComponent],
            Dict[str, Any]
        ],
        component_policy: Union[
            BasePolicy,
            BasePolicyComponent,
            Callable[..., BasePolicyComponent],
            Dict[str, Any]
        ],
        component_logger: Union[
            str,
            BaseLogger,
            LoggerComponent,
            Callable[..., LoggerComponent],
            Dict[str, Any]
        ],
        component_trainer: Union[
            TrainerComponent,
            Callable[..., TrainerComponent],
            Dict[str, Any]
        ],
        device: Optional[Union[str, int, torch.device]] = None,
        seed: Optional[int] = None,
        replay_buffer_kwargs: Optional[dict] = None,
        train_collector_kwargs: Optional[dict] = None,
        test_collector_kwargs: Optional[dict] = None,
        policy_kwargs: Optional[dict] = None,
        logger_kwargs: Optional[dict] = None,
        trainer_kwargs: Optional[dict] = None,
    ):
        super().__init__()

        self._state_objs.extend([
            'component_replay_buffer',
            'component_train_collector',
            'component_test_collector',
            'component_policy',
            'component_logger',
            'component_trainer',
        ])

        self._construct_agent(
            component_replay_buffer=component_replay_buffer,
            component_train_collector=component_train_collector,
            component_test_collector=component_test_collector,
            component_policy=component_policy,
            component_logger=component_logger,
            component_trainer=component_trainer,
            device=device,
            seed=seed,
            replay_buffer_kwargs=replay_buffer_kwargs,
            train_collector_kwargs=train_collector_kwargs,
            test_collector_kwargs=test_collector_kwargs,
            policy_kwargs=policy_kwargs,
            logger_kwargs=logger_kwargs,
            trainer_kwargs=trainer_kwargs,
       )

    def apply_seed(self, seed):
        if not seed is None:
            np.random.seed(seed)
            torch.manual_seed(seed)

    def _construct_agent(
       self, component_replay_buffer,
       component_train_collector, component_test_collector,
       component_policy, component_logger, component_trainer,
       device, seed, replay_buffer_kwargs, train_collector_kwargs,
       test_collector_kwargs, policy_kwargs, logger_kwargs, trainer_kwargs
    ):
        # create new empty dicts where necessary
        if replay_buffer_kwargs is None: replay_buffer_kwargs = {}
        if train_collector_kwargs is None: train_collector_kwargs = {}
        if test_collector_kwargs is None: test_collector_kwargs = {}
        if policy_kwargs is None: policy_kwargs = {}
        if logger_kwargs is None: logger_kwargs = {}
        if trainer_kwargs is None: trainer_kwargs = {}

        # set up attributes for all the components
        self.component_replay_buffer = None
        self.component_trainer = None
        self.component_train_collector = None
        self.component_test_collector = None
        self.component_replay_buffer = None
        self.component_policy = None
        self.component_logger = None

        # set up attributes for the passive agent interface
        self._passive_collector = None
        self._passive_trainer = None
        self._passive_col_gen = None

        # setup the device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.device = device
            
        # seed
        self.apply_seed(seed)

        # construct the trainer component
        self.component_trainer = construct_config_object(
            component_trainer, TrainerComponent,
            default_obj_constructor=TrainerComponent,
            obj_kwargs=dict(trainer_kwargs,
                agent=self,
                device=device,
                seed=seed
            )
        )

        # construct the train collector
        self.component_train_collector = construct_config_object(
            component_train_collector, CollectorComponent,
            default_obj_constructor=CollectorComponent,
            obj_kwargs=dict(train_collector_kwargs,
                agent=self,
                device=device,
                seed=seed,
                component_replay_buffer=component_replay_buffer
            )
        )

        # construct the test collector
        self.component_test_collector = construct_config_object(
            component_test_collector, CollectorComponent,
            default_obj_constructor=CollectorComponent,
            obj_kwargs=dict(test_collector_kwargs,
                agent=self,
                device=device,
                seed=seed
            )
        )

        # we only construct the replay buffer here if it was not already
        # constructed by the train collector
        if self.component_replay_buffer is None:
            self.component_replay_buffer = construct_config_object(
                component_replay_buffer, ReplayBufferComponent,
                default_obj_constructor=ReplayBufferComponent,
                obj_kwargs=dict(replay_buffer_kwargs,
                    agent=self,
                    device=device,
                    seed=seed
                )
            )
            
        # construct the policy
        self.component_policy = construct_config_object(
            component_policy, BasePolicyComponent,
            obj_kwargs=dict(policy_kwargs,
                agent=self,
                device=device,
                seed=seed
            )
        )
        
        # construct the logger
        self.component_logger = construct_config_object(
            component_logger, LoggerComponent,
            default_obj_constructor=LoggerComponent,
            obj_kwargs=dict(logger_kwargs,
                agent=self,
                device=device,
                seed=seed
            )
        )

        # run any additional component setup
        if not self.component_trainer is None:
            self.component_trainer.setup(agent=self, device=device, seed=seed)

        if not self.component_replay_buffer is None:
            self.component_replay_buffer.setup(agent=self, device=device, seed=seed)

        if not self.component_train_collector is None:
            self.component_train_collector.setup(agent=self, device=device, seed=seed)

        if not self.component_test_collector is None:
            self.component_test_collector.setup(agent=self, device=device, seed=seed)

        if not self.component_policy is None:
            self.component_policy.setup(agent=self, device=device, seed=seed)

        if not self.component_logger is None:
            self.component_logger.setup(agent=self, device=device, seed=seed)

    # properties
    @property
    def train_collector(self):
        return getattr(self.component_train_collector, 'collector', None)

    @train_collector.setter
    def train_collector(self, collector):
        if getattr(self.component_trainer, 'init_done', False):
            warnings.warn("Collectors should not be set after initialization: a trainer would already have been created with the old ones.")
        self.component_train_collector.collector = collector

    @property
    def test_collector(self):
        return getattr(self.component_test_collector, 'collector', None)

    @test_collector.setter
    def test_collector(self, collector):
        if getattr(self.component_trainer, 'init_done', False):
            warnings.warn("Collectors should not be set after initialization: a trainer would already have been created with the old ones.")
        self.component_test_collector.collector = collector

    @property
    def train_envs(self):
        try:
            return self.component_train_collector.collector.env
        except AttributeError:
            return None

    @property
    def test_envs(self):
        try:
            return self.component_test_collector.collector.env
        except AttributeError:
            return None

    def get_observation_space(self, observation_space=None):
        """Returns the observation space from the train collector or None if
        a train collector is not available.

        Optionally, an observation_space argument can be passed: the function
        is going to check if it is None; if not it is going to be returned
        instead of the collector's observation space.
        """
        if observation_space is None:
            if self.component_train_collector is None:
                return None
            else:
                return self.component_train_collector.observation_space

        return observation_space

    def get_observation_shape(self, observation_spec=None):
        """Returns the shape of the observation space from the train collector
        or None if a train collector is not available.

        Optionally, an observation_spec argument can be passed: the function
        is going to check if it is None; if not, it is going to be used
        in place of the collector's observation space. If it is a gym space,
        its shape is going to be extracted; if not, it is going to be returned
        directly. 
        """
        if observation_spec is None:
            if self.component_train_collector is None:
                return None
            else:
                return self.component_train_collector.observation_shape
        
        if isinstance(observation_spec, gym.spaces.Space):
            return extract_shape(observation_spec)
        else:
            return observation_spec

    def get_action_space(self, action_space):
        """Returns the action space from the train collector or None if
        a train collector is not available.

        Optionally, an action_space argument can be passed: the function
        is going to check if it is None; if not it is going to be returned
        instead of the collector's action space.
        """
        if action_space is None:
            if self.component_train_collector is None:
                return None
            else:
                return self.component_train_collector.action_space

        return action_space

    def get_action_shape(self, action_spec=None):
        """Returns the shape of the action space from the train collector
        or None if a train collector is not available.

        Optionally, an action_spec argument can be passed: the function
        is going to check if it is None; if not, it is going to be used
        in place of the collector's action space. If it is a gym space,
        its shape is going to be extracted; if not, it is going to be returned
        directly. 
        """
        if action_spec is None:
            if self.component_train_collector is None:
                return None
            else:
                return self.component_train_collector.action_shape
        
        if isinstance(action_spec, gym.spaces.Space):
            return extract_shape(action_spec)
        else:
            return action_spec

    @property
    def observation_space(self):
        return self.get_observation_space()

    @property
    def action_space(self):
        return self.get_action_space()

    @property
    def observation_shape(self):
        return self.get_observation_shape()

    @property
    def action_shape(self):
        return self.get_action_shape()

    @property
    def env_step(self):
        return self.component_logger.env_step

    @property
    def gradient_step(self):
        return self.component_logger.gradient_step

    @property
    def epoch(self):
        return self.component_logger.epoch

    @property
    def log_path(self):
        return self.component_logger.log_path

    @property
    def policy(self):
        if self.component_policy is None:
            return None
        else:
            return self.component_policy.policy

    @property
    def logger(self):
        if self.component_logger is None:
            return None
        else:
            return self.component_logger.logger

    @property
    def buffer(self):
        if self.component_replay_buffer is None:
            return None
        else:
            return self.component_replay_buffer.replay_buffer

    @property
    def replay_buffer(self):
        if self.component_replay_buffer is None:
            return None
        else:
            return self.component_replay_buffer.replay_buffer

    @property
    def train_callbacks(self):
        return self.component_trainer.train_callbacks

    @property
    def test_callbacks(self):
        return self.component_trainer.test_callbacks

    @property
    def save_best_callbacks(self):
        return self.component_trainer.save_best_callbacks

    @property
    def save_checkpoint_callbacks(self):
        return self.component_trainer.save_checkpoint_callbacks

    # the interface
    def reset_progress_counters(self):
        self.logger.reset_progress_counters()

    def make_trainer(self, **kwargs) -> BaseTrainer:
        """Creates a trainer. The keyword arguments (if any) are used to
        override the arguments originally passed to the trainer.
        Most notably, perhaps, you can override the:
            * ``max_epoch``;
            * ``batch_size``;
            * ``step_per_epoch``;
            * ``step_per_collect``;
            * ...

        Returns:
            A trainer.
        """
        return self.component_trainer.make_trainer(agent=self, **kwargs)
       
    def train(self, **kwargs) -> Dict[str, Union[float, str]]:
        """Runs training. The keyword arguments (if any) are used to
        override the arguments originally passed to the trainer.
        Most notably, perhaps, you can override the:
            * ``max_epoch``;
            * ``batch_size``;
            * ``step_per_epoch``;
            * ``step_per_collect``;
            * ...

        Returns:
            dict: See :func:`~tianshou.trainer.gather_info`.
        """
        return self.component_trainer.make_trainer(agent=self, **kwargs).run()
        
    def test(self,
        render: Optional[float] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Runs testing and returns the results from the test collector.

        Args:
            render (float, optional): The sleep time between rendering
                consecutive frames. Default to None (no rendering).
            seed (int, optional): The numeric seed used to seed the random
                number generators (``np.random``, ``torch.manual_seed``,
                ``train_envs.seed``, ``test_envs.seed``). Defaults to ``None``
                in which case no seeding is done.

        Returns:
            Dict[str, Any]: The results from the test collector.
        """
        # copy the keyword args
        kwargs = kwargs.copy()
        if not seed is None: self._apply_seed(seed)

        # the test_fn
        test_fn = kwargs.pop("test_fn", None)
        if test_fn is None:
            test_fn = self.component_trainer._test_fn

        if not test_fn is None:
            test_fn(self, self.logger.epoch, self.logger.env_step)   

        self.policy.eval()

        # episode per test
        episode_per_test = kwargs.pop("episode_per_test", None)

        if episode_per_test is None:
            episode_per_test = self.component_trainer.episode_per_test

        test_envs = self.test_envs
        if episode_per_test is None and not test_envs is None:
            episode_per_test = len(test_envs)

        self.test_collector.reset()
        return self.test_collector.collect(
            n_episode=episode_per_test, render=render, **kwargs
        )

    # the passive agent interface
    def init_passive_training(self,
        num_envs: int = 1,
        policy: Optional[BasePolicy] = None,
        buffer: Optional[ReplayBuffer] = None,
        preprocess_fn: Optional[Callable[..., Batch]] = None,
        exploration_noise: bool = False,
        **kwargs
    ):
        """Initializes the passive training. The keyword arguments (if any)
        are used to override default trainer arguments.
        """

        if policy is None: policy = self.policy
        if buffer is None: buffer = self.buffer

        self._passive_collector = PassiveCollector(
            policy=policy, buffer=buffer, preprocess_fn=preprocess_fn,
            num_envs=num_envs, exploration_noise=exploration_noise,
        )

        trainer_kwargs = dict(
            train_collector=self._passive_collector,
            test_collector=None,
            max_epoch=float('inf')
        )

        trainer_kwargs.update(kwargs)
        trainer_kwargs["policy"] = policy
        trainer_kwargs["buffer"] = buffer

        self._passive_trainer = StepWiseTrainer(self.make_trainer(**trainer_kwargs))
        self._passive_col_gen = None

    def finish_passive_training(self):
        """Finishes passive training by throwing away the passive collector
        and trainer.
        """
        self._passive_collector = None
        self._passive_trainer = None
        self._passive_col_gen = None

    def act(
        self,
        obs: Union[Batch, np.ndarray, torch.Tensor],
        done: Optional[Union[Batch, np.ndarray, torch.Tensor]],
        state: Optional[Batch],
        no_grad: bool = True,
        return_info: bool = False,
    ) -> Tuple[Batch, Batch]:
        return self._passive_collector.act(
            obs=obs, done=done, state=state, no_grad=no_grad,
            return_info=return_info
        )

    def observe_transition(
        self,
        data: Batch,
        env_ids: Optional[Union[np.ndarray, List[int]]] = None,
        step_per_collect: Optional[int] = None,
        episode_per_collect: Optional[int] = None,
    ) -> None:
        passive_trainer = self._passive_trainer

        if step_per_collect is None:
            step_per_collect = passive_trainer.step_per_collect

        if episode_per_collect is None:
            episode_per_collect = passive_trainer.episode_per_collect

        if self._passive_col_gen is None:
            self._passive_col_gen = self._passive_collector.make_collect(
                n_step=step_per_collect, n_episode=episode_per_collect
            )
            next(self._passive_col_gen)

        self._passive_collector.observe_transition(data=data, env_ids=env_ids)
        ret = next(self._passive_col_gen)

        # we only do training once enough data has been collected
        if not ret is None:
            self._passive_col_gen = None

            try:
                next(passive_trainer)
            except StopIteration:
                raise StopIteration("The passive trainer has stopped.")
            
class Agent(ComponentAgent):
    """An agent class built upon ComponentAgent: it only adds convenient
    agent-level arguments for some component parameters.

    Args:
        task_name (str): The name of the ``gym`` environment; by default,
            environments are constructed using ``gym.make``. To override
            this behaviour, supply a ``task`` argument: a callable that
            constructs your ``gym`` environment.
        max_epoch (int, optional): The maximum number of epochs for training.
            The training process might be finished before reaching
            ``max_epoch`` if the stop criterion returns ``True``; this
            behaviour can be overriden using the ``stop_criterion`` argument.
        train_envs (Union[int, List[Union[gym.Env, Callable[[], gym.Env]]], BaseVectorEnv], optional):
            A spec used to construct training environments. One of:
                * None;
                * The number of environments to construct;
                * A list of Gym environments / callables that construct
                  Gym environments;
                * A BaseVectorEnv instance.
            When the environments are constructed automatically, it is using
            the class specified through the train_env_class argument.
        test_envs (Union[int, List[Union[gym.Env, Callable[[], gym.Env]]], BaseVectorEnv], optional):
            A spec used to construct testing environments. One of:
                * None;
                * The number of environments to construct;
                * A list of Gym environments / callables that construct
                  Gym environments;
                * A BaseVectorEnv instance.
            When the environments are constructed automatically, it is using
            the class specified through the train_env_class argument.
        replay_buffer (Union[int, ReplayBuffer, Callable[[int], ReplayBuffer]], optional):
            The replay buffer to be used for experience collection. Note
            that in Tianshou replay buffers are used by both offline and
            online methods. The replay beffue spec can be one of:
                * int: the size of the replay buffer;
                * a replay buffer instance;
                * a BaseReplayBufferComponent instance;
                * a callable that takes creates a BaseReplayBufferComponent;
                * a config dictionary in the construct_config_object format;
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
        exploration_noise_train (bool, optional): Determines whether, during
            training, the action needs to be modified with the corresponding
            policy's exploration noise. If so, ``policy.exploration_noise(act, batch)``
            will be called automatically to add the exploration noise into the action.
            This is only used unless a pre-constructed collector is supplied.
        exploration_noise_test (bool, optional): Determines whether, during
            testing, the action needs to be modified with the corresponding
            policy's exploration noise. If so, ``policy.exploration_noise(act, batch)``
            will be called automatically to add the exploration noise into the action.
            This is only used unless a pre-constructed collector is supplied.
        episode_per_test (int, optional): The number of episodes for one
            policy evaluation.
        train_env_class (Type[BaseVectorEnv], optional): The vector environment
            used to represent the collection of training environments. See
            also ``train_envs``.
        test_env_class (Type[BaseVectorEnv], optional): The vector environment
            used to represent the collection of testing environments. See
            also ``test_envs``.
        logger (Union[str, BaseLogger, LoggerComponent, Callable[...,   
            LoggerComponent], Dict[str, Any], ], optional): The logger to use.
            The spec is passed to construct_config_object.
            
            If a string is passed, it is interpreted as log_dir: the directory
            that logs should go under. A log_path (the full path to the logs)
            is constructed by appending the name of the environment and the
            name of the agent to it as subdirectories.

            None is interpreted as log_dir = 'log'.

        task (Callable[[], gym.Env], optional): A callable used to
            construct training and testing environments. Defaults to None
            in which case evironments are constructed using
            gym.make(task_name).
        test_task (Callable[[], gym.Env], optional): A callable used to
            construct testing environments. Defaults to None in which
            case testing environments are constructed in the same way
            as training environments.

        train_callbacks (Union[
            List[CallbackType],
            Callable[..., Union[CallbackType, List[CallbackType]]],
            Dict[str, Any]
        ], optional): A list of callbacks invoked at the beginning of each
            training step. The signature of the callbacks is
            ``f(epoch: int, env_step: int, gradient_step: int, agent: Agent) -> None``.

            Optionally, save callbacks can be constructed using
            a construct_config_object spec.

        test_callbacks (Union[
            List[CallbackType],
            Callable[..., Union[CallbackType, List[CallbackType]]],
            Dict[str, Any]
        ], optional): A list of callbacks invoked at the beginning of each
            testing step. The signature of the callbacks is
            ``f(epoch: int, env_step: int, gradient_step: int, agent: Agent) -> None``.

            Optionally, save callbacks can be constructed using
            a construct_config_object spec.

        save_best_callbacks (Union[
            str,
            List[CallbackType],
            Callable[..., Union[CallbackType, List[CallbackType]]],
            Dict[str, Any]
        ], optional): A list of callbacks invoked every time the undiscounted
            average mean reward in evaluation phase gets better.

            The signature of the callbacks is
            ``f(epoch: int, env_step: int, gradient_step: int, agent: Agent) -> None``.

            Optionally, a callback can be constructed automatically using
            a construct_config_object spec.
            
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
            ``f(epoch: int, env_step: int, gradient_step: int, agent: Agent) -> None``.

            Optionally, a save callback can be constructed automatically using
            a construct_config_object spec. Unless otherwise specified, the
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

        verbose (bool): whether to print information when training.
            Defaults to True.

        component_replay_buffer (Union[int, ReplayBuffer, Callable[[int], ReplayBuffer]], optional):
            This is the same as replay_buffer (when component_replay_buffer
            is specified, it is used in place of replay_buffer and the value
            of replay_buffer is ignored).

        component_train_collector (Union[
            Collector,
            CollectorComponent,
            Callable[..., CollectorComponent],
            Dict[str, Any]
        ]):
            The component responsible for setting up the training collector.
            See CollectorComponent and agent presets for more details about
            the arguments and ways to construct collectors.

        component_test_collector (Union[
            Collector,
            CollectorComponent,
            Callable[..., CollectorComponent],
            Dict[str, Any]
        ]):
            The component responsible for setting up the testing collector.
            See CollectorComponent and agent presets for more details about
            the arguments and ways to construct collectors.

        component_policy (Union[
            Union[
            BasePolicy,
            BasePolicyComponent,
            Callable[..., BasePolicyComponent],
            Dict[str, Any]
        ]):
            The component responsible for setting up the policy. See
            agent presets for more details about the arguments and ways to construct policies.
        
        component_logger (Union[
            str,
            BaseLogger,
            LoggerComponent,
            Callable[..., LoggerComponent],
            Dict[str, Any]
        ]):
            This is the same as logger (when component_logger is specified,
             it is used in place of logger and the value of logger is ignored).

        component_trainer (Union[
            TrainerComponent,
            Callable[..., TrainerComponent],
            Dict[str, Any]
        ]):
            The component responsible for setting up the trainer. See
            TrainerComponent and agent presets for more details about the
            arguments and ways to construct trainers.

        reward_metric: a function with signature
            ``f(rewards: np.ndarray with shape (num_episode, agent_num)) -> np.ndarray
            with shape (num_episode,)``, used in multi-agent RL. We need to return a
            single scalar for each episode's result to monitor training in the
            multi-agent RL setting. This function specifies what is the desired metric,
            e.g., the reward of agent 1 or the average reward over all agents.

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

        device (Union[str, int, torch.device], optional): The PyTorch device
            to be used by PyTorch tensors and networks.

        seed (int, optional): The numeric seed used to seed the random
            number generators (``np.random``, ``torch.manual_seed``,
            ``train_envs.seed``, ``test_envs.seed``). Defaults to ``None``
            in which case no seeding is done.

        replay_buffer_kwargs (dict, optional): Additional keyword arguments to
            be passed to construct_config_object when constructing the replay
            buffer component.
        train_collector_kwargs (dict, optional): Additional keyword arguments to
            be passed to construct_config_object when constructing the train
            collector component.
        test_collector_kwargs (dict, optional): Additional keyword arguments to
            be passed to construct_config_object when constructing the test
            collector component.
        policy_kwargs (dict, optional): Additional keyword arguments to be
            passed to construct_config_object when constructing the policy
            component.
        logger_kwargs (dict, optional): Additional keyword arguments to be
            passed to construct_config_object when constructing the logger
            component.
        trainer_kwargs (dict, optional): Additional keyword arguments to be
            passed to construct_config_object when constructing the trainer
            component.

        Any additional keyword arguments are passed to the policy
        construction method (``_setup_policy(self, **kwargs)``).
    """
    def __init__(self,
        task_name: str,
        max_epoch: int = 10,
        train_envs: Optional[Union[int, List[Union[gym.Env, Callable[[], gym.Env]]], BaseVectorEnv]] = None,
        test_envs: Optional[Union[int, List[Union[gym.Env, Callable[[], gym.Env]]], BaseVectorEnv]] = None,
        replay_buffer: Union[
            int,
            ReplayBuffer,
            BaseReplayBufferComponent,
            Callable[..., BaseReplayBufferComponent],
            Dict[str, Any]
        ] = 1000000,
        step_per_epoch: int = 10000,
        step_per_collect: Optional[int] = None,
        update_per_collect: Optional[float] = 1.,
        update_per_step: Optional[float] = None,
        exploration_noise_train: bool = True,
        exploration_noise_test: bool = True,
        episode_per_test: Optional[int] = None,
        train_env_class: Optional[Type[BaseVectorEnv]] = None,
        test_env_class: Optional[Type[BaseVectorEnv]] = None,
        logger: Union[
            str,
            BaseLogger,
            LoggerComponent,
            Callable[..., LoggerComponent],
            Dict[str, Any]
        ] = 'auto',
        task: Optional[Callable[[], gym.Env]] = None,
        test_task: Optional[Callable[[], gym.Env]] = None,
        train_callbacks: Optional[Union[
            List[CallbackType],
            Callable[..., Union[CallbackType, List[CallbackType]]],
            Dict[str, Any]
        ]] = None,
        test_callbacks: Optional[Union[
            List[CallbackType],
            Callable[..., Union[CallbackType, List[CallbackType]]],
            Dict[str, Any]
        ]] = None,
        save_best_callbacks: Optional[Union[
            str,
            List[CallbackType],
            Callable[..., Union[CallbackType, List[CallbackType]]],
            Dict[str, Any]
        ]] = None,
        save_checkpoint_callbacks: Optional[Union[
            str,
            List[CallbackType],
            Callable[..., Union[CallbackType, List[CallbackType]]],
            Dict[str, Any]
        ]] = None,
        stop_criterion: Optional[
            Union[
                bool,
                str,
                Callable[[float, Union[float, None], 'ComponentAgent'], float]
            ]
        ] = None,
        verbose: bool = True,
        component_replay_buffer: Union[
            int,
            ReplayBuffer,
            BaseReplayBufferComponent,
            Callable[..., BaseReplayBufferComponent],
            Dict[str, Any]
        ] = None,
        component_train_collector: Union[
            Collector,
            CollectorComponent,
            Callable[..., CollectorComponent],
            Dict[str, Any]
        ] = None,
        component_test_collector: Union[
            Collector,
            CollectorComponent,
            Callable[..., CollectorComponent],
            Dict[str, Any]
        ] = None,
        component_policy: Union[
            BasePolicy,
            BasePolicyComponent,
            Callable[..., BasePolicyComponent],
            Dict[str, Any]
        ] = None,
        component_logger: Union[
            str,
            BaseLogger,
            LoggerComponent,
            Callable[..., LoggerComponent],
            Dict[str, Any]
        ] = None,
        component_trainer: Union[
            TrainerComponent,
            Callable[..., TrainerComponent],
            Dict[str, Any]
        ] = None,
        reward_metric: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        batch_size: int = 128,
        repeat_per_collect: Optional[int] = None,
        update_per_epoch: Optional[int] = None,
        episode_per_collect: Optional[int] = None,
        test_in_train: bool = False,
        prefill_steps: int = 0,
        resume_from_log: bool = True,
        reward_threshold: Optional[float] = None,
        device: Optional[Union[str, int, torch.device]] = None,
        seed: Optional[int] = None,
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
        replay_buffer_kwargs: Optional[dict] = None,
        train_collector_kwargs: Optional[dict] = None,
        test_collector_kwargs: Optional[dict] = None,
        policy_kwargs: Optional[dict] = None,
        logger_kwargs: Optional[dict] = None,
        trainer_kwargs: Optional[dict] = None,
        **kwargs
    ):
        # create new empty dicts where necessary
        if replay_buffer_kwargs is None: replay_buffer_kwargs = {}
        if train_collector_kwargs is None: train_collector_kwargs = {}
        if test_collector_kwargs is None: test_collector_kwargs = {}
        if policy_kwargs is None: policy_kwargs = {}
        if logger_kwargs is None: logger_kwargs = {}
        if trainer_kwargs is None: trainer_kwargs = {}

        # aliases
        if component_train_collector is None:
            component_train_collector = train_collector

        if component_test_collector is None:
            component_test_collector = test_collector

        # resolve env constructors
        train_task, test_task = self._resolve_tasks(task, test_task, task_name)

        # replay buffer
        if component_replay_buffer is None:
            component_replay_buffer = replay_buffer

        replay_buffer_kwargs = dict(dict(), **replay_buffer_kwargs)

        # train collector
        train_collector_kwargs = dict(dict(
            task_name=task_name,
            task=train_task,
            env=train_envs,
            exploration_noise=exploration_noise_train,
            env_class=train_env_class,

        ), **train_collector_kwargs)

        # test collector
        test_collector_kwargs = dict(dict(
            task_name=task_name,
            task=test_task,
            env=test_envs,
            exploration_noise=exploration_noise_test,
            env_class=test_env_class

        ), **test_collector_kwargs)

        # policy
        policy_kwargs = dict(dict(
            max_epoch=max_epoch,
        ), **policy_kwargs)
        
        policy_kwargs.update(kwargs)
        
        # logger
        if component_logger is None:
            component_logger = logger

        logger_kwargs = dict(dict(
            task_name=task_name
        ), **logger_kwargs)

        # trainer
        trainer_kwargs = dict(dict(
            max_epoch=max_epoch,
            step_per_epoch=step_per_epoch,
            step_per_collect=step_per_collect,
            update_per_collect=update_per_collect,
            update_per_step=update_per_step,
            episode_per_test=episode_per_test,
            train_callbacks=train_callbacks,
            test_callbacks=test_callbacks,
            save_best_callbacks=save_best_callbacks,
            save_checkpoint_callbacks=save_checkpoint_callbacks,
            stop_criterion=stop_criterion,
            verbose=verbose,
            reward_metric=reward_metric,
            batch_size=batch_size,
            repeat_per_collect=repeat_per_collect,
            update_per_epoch=update_per_epoch,
            episode_per_collect=episode_per_collect,
            test_in_train=test_in_train,
            resume_from_log=resume_from_log,
            reward_threshold=reward_threshold,
            prefill_steps=prefill_steps
        ), **trainer_kwargs)

        super().__init__(
            component_replay_buffer=component_replay_buffer,
            component_train_collector=component_train_collector,
            component_test_collector=component_test_collector,
            component_policy=component_policy,
            component_logger=component_logger,
            component_trainer=component_trainer,
            device=device,
            seed=seed,
            replay_buffer_kwargs=replay_buffer_kwargs,
            train_collector_kwargs=train_collector_kwargs,
            test_collector_kwargs=test_collector_kwargs,
            policy_kwargs=policy_kwargs,
            logger_kwargs=logger_kwargs,
            trainer_kwargs=trainer_kwargs
        )

    def _resolve_tasks(self, train_task, test_task, task_name):
        if train_task is None:
            train_task = partial(gym.make, task_name)

        if test_task is None:
            test_task = train_task

        return train_task, test_task
