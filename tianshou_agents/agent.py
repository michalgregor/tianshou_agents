from tianshou.data import ReplayBuffer, Collector
from tianshou.env import BaseVectorEnv
from tianshou.utils import BaseLogger
from tianshou.trainer import offpolicy_trainer
from .callback import SaveCallback
from .network import RLNetwork
from .utils import StateDictObject
from .components import AgentLogger, AgentCollector
from torch.optim import Optimizer
from functools import partial
from typing import List, Optional, Union, Callable, Type, Dict, Any
from gym.spaces import Tuple as GymTuple
from numbers import Number
import numpy as np
import torch
import gym
import abc

def resolve_tasks(train_task, test_task, task_name):
    if train_task is None:
        train_task = partial(gym.make, task_name)

    if test_task is None:
        test_task = train_task

    return train_task, test_task

class Agent(StateDictObject):
    def __init__(
        self, task_name: str, method_name: str,
        max_epoch: int = 10,
        train_envs: Union[int, List[Union[gym.Env, Callable[[], gym.Env]]], BaseVectorEnv] = 1,
        test_envs: Union[int, List[Union[gym.Env, Callable[[], gym.Env]]], BaseVectorEnv] = 1,
        replay_buffer: Union[int, ReplayBuffer, Callable[[int], ReplayBuffer]] = 1000000,
        step_per_epoch: int = 10000,
        step_per_collect: Optional[int] = None,
        update_per_collect: Optional[float] = 1.,
        exploration_noise_train: bool = True,
        exploration_noise_test: bool = True,
        train_env_class: Optional[Type[BaseVectorEnv]] = None,
        test_env_class: Optional[Type[BaseVectorEnv]] = None,
        episode_per_test: Optional[int] = None,
        train_callbacks: Optional[List[Callable[[int, int, int, 'Agent'], None]]] = None,
        test_callbacks: Optional[List[Callable[[int, int, int, 'Agent'], None]]] = None,
        save_callbacks: Optional[List[Callable[[int, int, int, 'Agent'], None]]] = None,
        save_checkpoint_callbacks: Optional[List[Callable[[int, int, int, 'Agent'], None]]] = None,
        train_collector: Optional[Union[Collector, Callable[..., Collector]]] = None,
        test_collector: Optional[Union[Collector, Callable[..., Collector]]] = None,
        seed: int = None,
        logger: Optional[Union[str, Dict[str, Any], BaseLogger]] = "log",
        device: Union[str, int, torch.device] = "cpu",
        task: Optional[Callable[[], gym.Env]] = None,
        test_task: Optional[Callable[[], gym.Env]] = None,
        stop_criterion: Union[bool, Callable[[float, Union[float, None], 'Agent'], float]] = False,
        **policy_kwargs
    ):
        """The base class of Agents, which provides the common functionality
        such as logging, callbacks, environment construction, etc.

        To subclass this, you need to implement at least the following:
            * ``_setup_policy(self, **kwargs)``;
            * ``train(self, **kwargs)``;
            * add objects whose state_dict should be included in the agent's
              state_dict() [i.e. their state changes as the agent trains and
              should be saved when checkpointing the agent] to the
              self._state_objs list (by appending their attribute name or
              nested attribute name as a string) - or you can also override
              state_dict() in the subclass, if necessary;

        Most often, though, ``train`` would be inhereted through subclasses
        such as ``OffPolicyAgent`` or ``OnPolicyAgent``.

        Args:
            task_name (str): The name of the ``gym`` environment; by default,
                environments are constructed using ``gym.make``. To override
                this behaviour, supply a ``task`` argument: a callable that
                constructs your ``gym`` environment.
            method_name (str): The name of the RL method that the agent
                implements. This is used e.g. to determine the default
                logging path.
            max_epoch (int, optional): The maximum number of epochs for training. The
                training process might be finished before reaching ``max_epoch``
                if the stop criterion returns ``True``; this behaviour can be
                overriden using the ``stop_criterion`` argument.
            train_envs (Union[int, List[Union[gym.Env, Callable[[], gym.Env]]], BaseVectorEnv], optional):
                Either a collection of environment instances / callables that return environment instances or the number of environments to be used (the collection of
                environments is constructed automatically using
                ``train_env_class``); these environments are used to train the agent.
            test_envs (Union[int, List[Union[gym.Env, Callable[[], gym.Env]]], BaseVectorEnv], optional):
                Either a collection of environment instances / callables that return environment instances or
                the number of environments to be used (the collection of
                environments is constructed automatically using
                ``test_env_class``); these environments are used for testing the agent.
            replay_buffer (Union[int, ReplayBuffer, Callable[[int], ReplayBuffer]], optional):
                The replay buffer to be used for experience collection. Note
                that in Tianshou replay buffers are used by both offline and
                online methods. Here, ``replay_buffer`` is either a ``ReplayBuffer``
                instance or ``callable(num_train_envs)`` that returns a
                ``ReplayBuffer`` instance.
            step_per_epoch (int, optional): The number of transitions collected per epoch.
            step_per_collect (int, optional): The number of transitions a collector
                would collect before the network update, i.e., trainer will
                collect ``step_per_collect`` transitions and do some policy
                network update repeatly in each epoch. Defaults to ``None``, 
                which means that ``step_per_collect`` is the same as the
                number of training environments.
            update_per_collect (float, optional): The number of times the policy
                network will be updated for each collect (collection of
                experience from the environments).
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
            train_env_class (Type[BaseVectorEnv], optional): The vector environment
                used to represent the collection of training environments. See
                also ``train_envs``.
            test_env_class (Type[BaseVectorEnv], optional): The vector environment
                used to represent the collection of testing environments. See
                also ``test_envs``.
            episode_per_test (int, optional): The number of episodes for one policy evaluation.
            train_callbacks (List[Callable[[int, int, int, Agent], None]]): A list of
                callbacks invoked at the beginning of each training step.
                The signature of the callbacks is
                ``f(epoch: int, env_step: int, gradient_step: int, agent: Agent) -> None``.
            test_callbacks (List[Callable[[int, int, int, Agent], None]]): A list of
                callbacks invoked at the beginning of each testing step.
                The signature of the callbacks is
                ``f(epoch: int, env_step: int, gradient_step: int, agent: Agent) -> None``.
            save_callbacks (List[Callable[[int, int, int, Agent], None]]): A list of
                callbacks invoked every time the undiscounted average mean reward in
                evaluation phase gets better.

                The signature of the callbacks is
                ``f(epoch: int, env_step: int, gradient_step: int, agent: Agent) -> None``.

                If None, a callback that makes checkpoints is constructed
                automatically. To disable checkpointing altogether, pass an
                empty list.
            save_checkpoint_callbacks (List[Callable[[int, int, int, Agent], None]]):
                A list of callbacks invoked after every step of training.

                The signature of the callbacks is
                ``f(epoch: int, env_step: int, gradient_step: int, agent: Agent) -> None``.

                If None, an empty list is created.
            train_collector (Optional[Union[Collector, Callable[..., Collector]]], optional): 
                The collector used to collect experience during training. It can
                either be a ``Collector`` instance or ``callable(policy, envs, buffer, exploration_noise)``.
                Defaults to ``None``, which means that a ``Collector`` is
                constructed automatically. See also ``exploration_noise_train``.
            test_collector (Optional[Union[Collector, Callable[..., Collector]]], optional): 
                The collector used to collect experience during testing. It can
                either be a ``Collector`` instance or ``callable(policy, envs, buffer, exploration_noise)``.
                Defaults to ``None``, which means that a ``Collector`` is
                constructed automatically. See also ``exploration_noise_test``.
            seed (int, optional): The numeric seed used to seed the random
                number generators (``np.random``, ``torch.manual_seed``,
                ``train_envs.seed``, ``test_envs.seed``). Defaults to ``None``
                in which case no seeding is done.
            logger (Union[str, Dict[str, Any], BaseLogger], optional):
                The logger to use. If an instance of BaseLogger, the logger
                is used for logging directly.

                If a dictionary, it is interpreted as a set of keyword arguments
                for constructing a logger. In that case, the constructor is to
                be provided under key ``type``. If ``type`` is not present or
                is None, then a TensorboardLogger is constructed. In this case,
                a tensorboard writer is set up automatically, unless already
                present among the keyword arguments.

                When auto-constructing a writer, the path defaults to "log".
                If a ``log_path`` key is provided, the writer is set up with
                that path. If ``log_path`` is None and ``log_dir`` is provided,
                then the path is constructed automatically by appending the
                name of the environment and the name of the agent to it as
                subdirectories.

                If a string is provided, it is interpreted as ``log_dir``.

                None is interpreted as ``log_dir = 'log'``.
            device (Union[str, int, torch.device], optional): The PyTorch device
                to be used by PyTorch tensors and networks.
            task (Callable[[], gym.Env], optional): A callable used to
                construct training and testing environments. Defaults to None
                in which case evironments are constructed using ``gym.make(task_name)``.
            test_task (Callable[[], gym.Env], optional): A callable used to
                construct testing environments. Defaults to None in which
                case testing environments are constructed in the same way
                as training environments.
            stop_criterion (Union[bool, Callable[[float, Union[float, None], 'Agent'], float]]):
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

            Any additional keyword arguments are passed to the policy
            construction method (``_setup_policy(self, **kwargs)``).
        """
        super().__init__()

        self._device = device
        self.max_epoch = max_epoch
        self.step_per_epoch = step_per_epoch
        self.step_per_collect = step_per_collect
        self.update_per_collect = update_per_collect
        self.stop_criterion = stop_criterion
        self.train_callbacks = train_callbacks or []
        self.test_callbacks = test_callbacks or []
        self.save_callbacks = save_callbacks
        self.save_checkpoint_callbacks = save_checkpoint_callbacks

        self._state_objs.extend([
            'stop_criterion',
            'train_callbacks',
            'test_callbacks',
            'save_callbacks',
            'save_checkpoint_callbacks',
            'collector',
            'logger',
            'policy'
        ])

        # seed
        self._apply_seed(seed)

        # resolve env constructors
        train_task, test_task = resolve_tasks(task, test_task, task_name)

        # setup the collectors
        self.collector = self._setup_collector(
            train_collector, test_collector,
            train_task, test_task,
            train_env_class, test_env_class,
            train_envs, test_envs,
            exploration_noise_train, exploration_noise_test,
            replay_buffer,
            seed
        )

        # spaces
        self.observation_space = self.collector.train_envs.observation_space[0]
        self.action_space = self.collector.train_envs.action_space[0]

        if isinstance(self.observation_space, GymTuple):
            self.state_shape = (osp.shape or osp.n for osp in self.observation_space)
        else:
            self.state_shape = self.observation_space.shape or self.observation_space.n

        self.action_shape = self.action_space.shape or self.action_space.n

        try:
            self.reward_threshold = self.collector.train_envs.spec[0].reward_threshold
        except AttributeError:
            self.reward_threshold = None

        # episode per test
        if episode_per_test is None:
            self.episode_per_test = len(self.collector.test_envs)
        else:
            self.episode_per_test = episode_per_test

        self._setup_policy(**policy_kwargs)
        self.collector.train_collector.policy = self.policy
        self.collector.test_collector.policy = self.policy

        self.logger = self._setup_logger(logger, task_name, method_name, seed)

        if self.step_per_collect is None:
            self.step_per_collect = len(self.collector.train_envs)

        if self.save_callbacks is None:
            self.save_callbacks = [SaveCallback(self.logger.log_path)]

        if self.save_checkpoint_callbacks is None:
            self.save_checkpoint_callbacks = []

    @abc.abstractmethod
    def _setup_policy(self, **kwargs):
        """
        Performs setup for the policy, creating the ``self.policy`` attribute
        that is going to be passed to the trainer; optionally also exposes
        other related attributes.
        """
        raise NotImplementedError()

    @property
    def env_step(self):
        return self.logger.env_step

    @property
    def gradient_step(self):
        return self.logger.gradient_step

    @property
    def epoch(self):
        return self.logger.epoch

    @property
    def log_path(self):
        return self.logger.log_path

    def reset_progress_counters(self):
        self.logger.reset_progress_counters()

    @abc.abstractmethod
    def train(self, **kwargs):
        raise NotImplementedError()

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
        if not seed is None: self._apply_seed(seed)
        self._test_fn(self.logger.epoch, self.logger.env_step)
        self.policy.eval()
    
        self.collector.test_collector.reset()
        return self.collector.test_collector.collect(
            n_episode=self.episode_per_test, render=render, **kwargs
        )

    def _setup_logger(self, logger, task_name, method_name, seed):
        return AgentLogger(logger, task_name, method_name, seed)

    def _setup_collector(self,
        train_collector, test_collector,
        train_task, test_task,
        train_env_class, test_env_class,
        train_envs, test_envs,
        exploration_noise_train, exploration_noise_test,
        replay_buffer,
        seed
    ):
        return AgentCollector(
            train_collector, test_collector,
            train_task, test_task,
            train_env_class, test_env_class,
            train_envs, test_envs,
            exploration_noise_train, exploration_noise_test,
            replay_buffer,
            seed
        )

    def _save_fn(self, policy):
        for callback in self.save_callbacks:
            callback(self.logger.epoch, self.logger.env_step, self.logger.gradient_step, self)

    def _save_checkpoint_fn(self, epoch, env_step, gradient_step):
        for callback in self.save_checkpoint_callbacks:
            callback(self.logger.epoch, self.logger.env_step, self.logger.gradient_step, self)

    def _stop_fn(self, mean_rewards):
        if callable(self.stop_criterion):
            return self.stop_criterion(mean_rewards, self.reward_threshold, self)
        elif isinstance(self.stop_criterion, Number) and not isinstance(self.stop_criterion, bool):
            return mean_rewards >= self.stop_criterion
        elif self.stop_criterion and not self.reward_threshold is None:
            return mean_rewards >= self.reward_threshold
        else:
            return False

    def _train_fn(self, epoch, env_step):
        self.logger.epoch = epoch
        self.logger.env_step = env_step
        for callback in self.train_callbacks:
            callback(epoch, env_step, self.logger.gradient_step, self)

    def _test_fn(self, epoch, env_step):
        for callback in self.test_callbacks:
            callback(epoch, env_step, self.logger.gradient_step, self)

    def _apply_seed(self, seed):
        if not seed is None:
            np.random.seed(seed)
            torch.manual_seed(seed)

    def construct_rlnet(
        self, module, state_shape, action_shape, **kwargs
    ):
        if module is None:
            module = RLNetwork(
                state_shape, action_shape,
                device=self._device,
                **kwargs
            ).to(self._device)
        elif isinstance(module, torch.nn.Module):
            module = module.to(self._device)
        elif isinstance(module, dict):
            kwargs = kwargs.copy()
            kwargs.update(module)
            module = kwargs.pop("type", RLNetwork)
            module = module(
                state_shape, action_shape,
                device=self._device,
                **kwargs
            ).to(self._device)

        else:
            module = module(
                state_shape, action_shape,
                device=self._device,
                **kwargs
            ).to(self._device)

        return module

    def construct_optim(self, optim, model_params):
        if optim is None:
            optim = torch.optim.Adam(model_params)
        elif isinstance(optim, Optimizer):
            pass
        elif isinstance(optim, dict):
            kwargs = optim.copy()
            optim = kwargs.pop("type", torch.optim.Adam)
            optim = optim(model_params, **kwargs)
        else:
            optim = optim(model_params)

        return optim

class OffPolicyAgent(Agent):
    def __init__(self,
        batch_size: int = 128,
        prefill_steps: int = 0,
        test_in_train = False,
        **kwargs
    ):
        """The base agent class for off-policy agents.

        Args:
            batch_size (int): The batch size of the sample data, that is going
                to be fed into the policy network.
            prefill_steps (int): The number of transitions used to prefill
                the replay buffer with experience.
            eps_prefill (float): The epsilon (exploration rate) to use when
                prefilling the replay buffer with experience.

            Any additional keyword arguments are passed to the base class.
        """
        super().__init__(**kwargs)

        self.test_in_train = test_in_train
        self.batch_size = batch_size

        if prefill_steps is None:
            self.prefill_steps = self.batch_size * len(self.collector.train_envs)
        else:
            self.prefill_steps = prefill_steps

    def _init(self):
        # prefill the replay buffer
        if self.prefill_steps:
            self.collector.train_collector.collect(n_step=self.prefill_steps, random=True)

    def train(self, seed=None, **kwargs) -> Dict[str, Union[float, str]]:
        """Runs off-policy training. The keyword arguments (if any) are used
        to update the arguments passed to ``offpolicy_trainer``. Most notably,
        perhaps, you can override the:
            * ``max_epoch``;
            * ``batch_size``;
            * ``step_per_epoch``;
            * ``step_per_collect``;
            * ``episode_per_test``;
            * ``update_per_collect``;
            * ``logger``.

        Returns:
            dict: See :func:`~tianshou.trainer.gather_info`.
        """
        params = dict(
            policy=self.policy,
            train_collector=self.collector.train_collector,
            test_collector=self.collector.test_collector,
            max_epoch=self.max_epoch,
            step_per_epoch=self.step_per_epoch,
            step_per_collect=self.step_per_collect,
            episode_per_test=self.episode_per_test,
            batch_size=self.batch_size,
            update_per_collect=self.update_per_collect,
            stop_fn=self._stop_fn,
            train_fn=self._train_fn,
            test_fn=self._test_fn,
            save_fn=self._save_fn,
            save_checkpoint_fn=self._save_checkpoint_fn,
            logger=self.logger,
            resume_from_log=True,
            test_in_train=self.test_in_train,
        )
        params.update(kwargs)
        update_per_collect = params.pop("update_per_collect")
        params["update_per_step"] = update_per_collect / params["step_per_collect"]

        # check whether training should actually start
        if self.logger.epoch and self.logger.epoch >= params["max_epoch"]: return

        # apply the seed
        self._apply_seed(seed)
        # if no training has been done yet
        if self.logger.env_step is None: self._init()

        self.policy.train()
        return offpolicy_trainer(**params)
