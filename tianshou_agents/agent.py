from .utils.state_dict import StateDictObject
from .utils.config_router import DefaultConfigRouter, BaseConfigRouter
from .components.replay_buffer import BaseReplayBufferComponent
from .components.collector import CollectorComponent
from .components.passive_interface import PassiveInterface
from .components.policy import BasePolicyComponent
from .components.logger import LoggerComponent
from .components.trainer import TrainerComponent
from tianshou.data import ReplayBuffer, Collector, Batch
from tianshou.policy import BasePolicy
from tianshou.trainer import BaseTrainer
from tianshou.utils.logger.base import BaseLogger

from typing import Optional, Union, Callable, Dict, Any, List, Tuple
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
        mode: bool = 'train'
    ) -> Tuple[Batch, Batch]:
        pass

    @abc.abstractmethod
    def observe_transition(
        self,
        data: Batch,
        env_ids: Optional[Union[np.ndarray, List[int]]] = None
    ) -> None:
        pass

    @abc.abstractmethod
    def train_mode(self):
        """Put the agent into train mode."""
        
    @abc.abstractmethod
    def eval_mode(self):
        """Put the agent into eval mode."""

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

    For convenience, some arguments can also be passed as agent-level arguments,
    which will be automatically routed to the individual components. For info
    about these arguments, see the DefaultConfigRouter class – or any other
    config router that you are using in its place. Note that the config
    router is usually supplied by the agent preset.
    
    Args:
        replay_buffer (Union[
            int,
            ReplayBuffer,
            Callable[[int], ReplayBuffer]
        ], optional):
            The replay buffer to be used for experience collection as 
            a ConfigBuilder spec. Note that in Tianshou replay buffers are
            used by both offline and online methods. The replay buffer spec
            can be one of:
                * int: the size of the replay buffer;
                * a replay buffer instance;
                * a BaseReplayBufferComponent instance;
                * a callable that takes creates a BaseReplayBufferComponent;
                * a config dictionary in the ConfigBuilder format;

        train_collector (Union[
            Collector,
            CollectorComponent,
            Callable[..., CollectorComponent],
            Dict[str, Any]
        ]):
            The component responsible for setting up the training collector
            as a ConfigBuilder spec. See CollectorComponent and agent presets
            for more details about the arguments and ways to construct
            collectors.

        test_collector (Union[
            Collector,
            CollectorComponent,
            Callable[..., CollectorComponent],
            Dict[str, Any]
        ]):
            The component responsible for setting up the test collector
            as a ConfigBuilder spec. See CollectorComponent and agent presets
            for more details about the arguments and ways to construct
            collectors.

        policy (Union[
            BasePolicy,
            BasePolicyComponent,
            Callable[..., BasePolicyComponent],
            Dict[str, Any]
        ]):
            The component responsible for setting up the policy as a 
            ConfigBuilder spec. See agent presets for more details about the
            arguments and ways to construct policies.

        logger (Union[
            str,
            BaseLogger,
            LoggerComponent,
            Callable[..., LoggerComponent],
            Dict[str, Any], 
        ], optional):
            The component responsible for setting up logging as a 
            ConfigBuilder spec.
            
            If a string is passed, it is interpreted as log_dir: the directory
            that logs should go under. A log_path (the full path to the logs)
            is constructed by appending the name of the environment and the
            name of the agent to it as subdirectories.

            An empty dict is interpreted as log_dir = 'log'.

        trainer (Union[
            TrainerComponent,
            Callable[..., TrainerComponent],
            Dict[str, Any]
        ]):
            The component responsible for setting up the trainer as a
            ConfigBuilder spec. See TrainerComponent and agent presets for more
            details about the arguments and ways to construct trainers.

        passive_interface (Union[
            PassiveInterface,
            Callable[..., PassiveInterface],
            Dict[str, Any]
        ]):
            The component that initializes the passive training interface
            specified as a ConfigBuilder spec. See PassiveInterface for more
            details about the arguments.

        device (Union[str, int, torch.device], optional): The PyTorch device
            to be used by PyTorch tensors and networks.

        seed (int, optional): The numeric seed used to seed the random
                number generators (``np.random``, ``torch.manual_seed``,
                ``train_envs.seed``, ``test_envs.seed``). Defaults to ``None``
                in which case no seeding is done.
    """
    def __init__(self,
        replay_buffer: Union[
            int,
            ReplayBuffer,
            BaseReplayBufferComponent,
            Callable[..., BaseReplayBufferComponent],
            Dict[str, Any]
        ],
        train_collector: Union[
            Collector,
            CollectorComponent,
            Callable[..., CollectorComponent],
            Dict[str, Any]
        ],
        test_collector: Union[
            Collector,
            CollectorComponent,
            Callable[..., CollectorComponent],
            Dict[str, Any]
        ],
        policy: Union[
            BasePolicy,
            BasePolicyComponent,
            Callable[..., BasePolicyComponent],
            Dict[str, Any]
        ],
        logger: Union[
            str,
            BaseLogger,
            LoggerComponent,
            Callable[..., LoggerComponent],
            Dict[str, Any]
        ],
        trainer: Union[
            TrainerComponent,
            Callable[..., TrainerComponent],
            Dict[str, Any]
        ],
        passive_interface: Union[
            PassiveInterface,
            Callable[..., PassiveInterface],
            Dict[str, Any]
        ] = None,
        device: Optional[Union[str, int, torch.device]] = None,
        seed: Optional[int] = None,
        config_router: Optional[BaseConfigRouter] = None
    ):
        super().__init__()

        self._state_objs.extend([
            'component_replay_buffer',
            'component_train_collector',
            'component_test_collector',
            'component_policy',
            'component_logger',
            'component_trainer',
            'component_passive_interface'
        ])

        self._construct_agent(
            replay_buffer=replay_buffer,
            train_collector=train_collector,
            test_collector=test_collector,
            policy=policy,
            logger=logger,
            trainer=trainer,
            passive_interface=passive_interface,
            device=device,
            seed=seed,
            config_router=config_router
       )

    def apply_seed(self, seed):
        if not seed is None:
            np.random.seed(seed)
            torch.manual_seed(seed)

    def _construct_agent(
       self, replay_buffer, train_collector, test_collector,
       policy, logger, trainer, passive_interface, device, seed,
       config_router
    ):
        self.config_router = config_router or DefaultConfigRouter()

        # set up attributes for all the components
        self.component_replay_buffer = None
        self.component_trainer = None
        self.component_train_collector = None
        self.component_test_collector = None
        self.component_replay_buffer = None
        self.component_policy = None
        self.component_logger = None
        self.component_passive_interface = None

        # setup the device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.device = device
            
        # seed
        self.apply_seed(seed)

        # construct the trainer component
        self.component_trainer = self.config_router.trainer_builder(
            trainer,
            default_kwargs=dict(
                agent=self,
                device=device,
                seed=seed
            )
        )
        
        # construct the train collector
        self.component_train_collector = self.config_router.collector_builder(
            train_collector,
            default_kwargs=dict(
                agent=self,
                device=device,
                seed=seed,
                replay_buffer=replay_buffer
            )
        )

        # construct the test collector
        self.component_test_collector = self.config_router.collector_builder(
            test_collector,
            default_kwargs=dict(
                agent=self,
                device=device,
                seed=seed
            )
        )

        # we only construct the replay buffer here if it was not already
        # constructed by the train collector
        if self.component_replay_buffer is None:
            self.component_replay_buffer = self.config_router.replay_buffer_builder(
                replay_buffer,
                default_kwargs=dict(
                    agent=self,
                    device=device,
                    seed=seed
                )
            )
            
        # construct the policy
        self.component_policy = self.config_router.policy_builder(
            policy,
            default_kwargs=dict(
                agent=self,
                device=device,
                seed=seed
            )
        )
        
        # construct the logger
        self.component_logger = self.config_router.logger_builder(
            logger,
            default_kwargs=dict(
                agent=self,
                device=device,
                seed=seed
            )
        )

        # construct the passive interface
        self.component_passive_interface = self.config_router.passive_interface_builder(
            passive_interface,
            default_kwargs=dict(
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

        if not self.component_passive_interface is None:
            self.component_passive_interface.setup(agent=self, device=device, seed=seed)

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
    def passive_collector(self):
        if not self.component_passive_interface is None:
            return getattr(self.component_passive_interface, 'collector', None)

    @property
    def passive_trainer(self):
        if not self.component_passive_interface is None:
            return getattr(self.component_passive_interface, 'trainer', None)

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

    @property
    def observation_space(self):
        return self.get_observation_space()

    @property
    def action_space(self):
        return self.get_action_space()

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
    def episode(self):
        return self.component_logger.episode 

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

    def act(
        self,
        obs: Union[Batch, np.ndarray, torch.Tensor],
        done: Optional[Union[Batch, np.ndarray, torch.Tensor]],
        state: Optional[Batch],
        no_grad: bool = True,
        return_info: bool = False,
        mode: bool = 'train',
        run_callbacks: bool = True
    ) -> Tuple[Batch, Batch]:
        """Calls upon the policy and returns the action along with the new
        state of the policy (only relevant for recurrent policies).

        Args:
            obs (Union[Batch, np.ndarray, torch.Tensor]): _description_
            done (Union[Batch, np.ndarray, torch.Tensor], optional): _description_
            state (Batch, optional): _description_
            no_grad (bool, optional): _description_. Defaults to True.
            return_info (bool, optional): _description_. Defaults to False.
            mode (bool, optional): The mode that the actor is to work in.
                One of 'train', 'eval' / 'test'. Defaults to 'train'.

        Returns:
            Tuple[Batch, Batch]: action batch, state batch
        """
        if mode == 'train':
            self.component_passive_interface.collector.policy.train()
            if run_callbacks: self.component_trainer.run_train_callbacks(self)

        elif mode == 'eval' or mode == 'test':
            self.component_passive_interface.collector.policy.eval()
            if run_callbacks: self.component_trainer.run_test_callbacks(self)

        else:
            raise ValueError("mode must be 'train' or 'eval' / 'test'.")

        return self.component_passive_interface.collector.compute_action(
            obs=obs, done=done, state=state, no_grad=no_grad,
            return_info=return_info
        )

    def observe_transition(
        self,
        data: Batch,
        env_ids: Optional[Union[np.ndarray, List[int]]] = None,
        step_per_collect: Optional[int] = None,
        episode_per_collect: Optional[int] = None,
        training_enabled: bool = True,
    ) -> None:
        passive_trainer = self.component_passive_interface.trainer
        passive_collector = self.component_passive_interface.collector

        if step_per_collect is None:
            step_per_collect = passive_trainer.step_per_collect

        if episode_per_collect is None:
            episode_per_collect = passive_trainer.episode_per_collect

        if not passive_collector.is_collecting:
            passive_collector.make_collect(
                n_step=step_per_collect, n_episode=episode_per_collect
            )

        ret = passive_collector.observe_transition(
            data=data, env_ids=env_ids
        )

        # we only do training once enough data has been collected
        if not ret is None and training_enabled:
            try:
                next(passive_trainer)
            except StopIteration:
                raise StopIteration("The passive trainer has stopped.")
