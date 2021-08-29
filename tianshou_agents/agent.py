from tianshou.data import ReplayBuffer, VectorReplayBuffer, Collector
from tianshou.env import BaseVectorEnv, DummyVectorEnv, SubprocVectorEnv
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import BasicLogger
from .network import RLNetwork
from torch.optim import Optimizer

from torch.utils.tensorboard import SummaryWriter
from typing import List, Optional, Union, Callable, Type, Dict, Any
from gym.spaces import Tuple as GymTuple
from numbers import Number
import numpy as np
import torch
import gym
import abc
import os

class Agent:
    def __init__(
        self, task_name: str, method_name: str,
        max_epoch: int = 10,
        train_envs: Union[int, BaseVectorEnv] = 1,
        test_envs: Union[int, BaseVectorEnv] = 1,
        replay_buffer: Union[int, ReplayBuffer, Callable[[int], ReplayBuffer]] = 1000000,
        step_per_epoch: int = 10000,
        step_per_collect: Optional[int] = None,
        update_per_step: Optional[float] = None,
        exploration_noise_train: bool = True,
        exploration_noise_test: bool = True,
        train_env_class: Optional[Type[BaseVectorEnv]] = None,
        test_env_class: Optional[Type[BaseVectorEnv]] = None,
        episode_per_test: Optional[int] = None,
        train_callbacks: Optional[List[Callable[[int, int], None]]] = None,
        test_callbacks: Optional[List[Callable[[int, int], None]]] = None,
        train_collector: Optional[Union[Collector, Callable[..., Collector]]] = None,
        test_collector: Optional[Union[Collector, Callable[..., Collector]]] = None,
        seed: int = None,
        logdir: str = "log",
        device: Union[str, int, torch.device] = "cpu",
        task: Optional[Callable[[], gym.Env]] = None,
        test_task: Optional[Callable[[], gym.Env]] = None,
        stop_criterion: Union[bool, Callable[[float, Union[float, None], 'Agent'], bool]] = True,
        **policy_kwargs
    ):
        """The base class of Agents, which provides the common functionality
        such as logging, callbacks, environment construction, etc.

        To subclass this, you need to implement at least the following:
            * ``_setup_policy(self, **kwargs)``;
            * ``train(self, **kwargs)``;

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
            train_envs (Union[int, BaseVectorEnv], optional): Either the
                collection of environment instances used to train the agent or
                the number of environments to be used (the collection of
                environments is constructed automatically using
                ``train_env_class``).
            test_envs (Union[int, BaseVectorEnv], optional): Either the
                collection of environment instances used to test the agent or
                the number of environments to be used (the collection of
                environments is constructed automatically using
                ``test_env_class``).
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
            update_per_step (float, optional): The number of times the policy network
                would be updated per transition after (``step_per_collect``)
                transitions are collected, e.g., if ``update_per_step`` set to
                0.3, and ``step_per_collect`` is 256, the policy will be updated
                ``round(256 * 0.3 = 76.8) = 77`` times after 256 transitions are
                collected by the collector. Defaults to ``None``, which means
                ``update_per_step = 1 / step_per_collect``.
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
            train_callbacks (List[Callable[[int, int], None]]): A list of
                callbacks invoked at the beginning of each training step.
                The signature of the callbacks is ``f(epoch: int, env_step: int) -> None``.
            test_callbacks (List[Callable[[int, int], None]]): A list of
                callbacks invoked at the beginning of each testing step.
                The signature of the callbacks is ``f(epoch: int, env_step: int) -> None``.
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
            logdir (str, optional): The path to the directory where logs will be kept.
            device (Union[str, int, torch.device], optional): The PyTorch device
                to be used by PyTorch tensors and networks.
            task (Callable[[], gym.Env], optional): A callable used to
                construct training and testing environments. Defaults to None
                in which case evironments are constructed using ``gym.make(task_name)``.
            test_task (Callable[[], gym.Env], optional): A callable used to
                construct testing environments. Defaults to None in which
                case testing environments are constructed in the same way
                as training environments.
            stop_criterion (Union[bool, Callable[[float, Union[float, None], 'Agent'], bool]]):
                The criterion used to stop training before ``max_epoch`` has
                been reached:
                    * If set to ``True``, training stops once the
                      mean reward reaches the environment's reward threshold;
                    * If set to ``False``, the stop criterion is disabled;
                    * If set to ``callable(mean_rewards, reward_threshold, agent)``,
                      the callable is used to determine whether training should
                      be stopped or not.

            Any additional keyword arguments are passed to the policy
            construction method (``_setup_policy(self, **kwargs)``).
        """

        self._inialized = False
        self._seed = seed
        self._device = device
        self.max_epoch = max_epoch
        self.step_per_epoch = step_per_epoch
        self.step_per_collect = step_per_collect
        self.update_per_step = update_per_step
        self.stop_criterion = stop_criterion
        self.train_callbacks = train_callbacks or []
        self.test_callbacks = test_callbacks or []
        self.epoch = 0
        self.env_step = 0

        # envs
        self._setup_envs(
            task_name, task, test_task,
            train_env_class, train_envs,
            test_env_class, test_envs
        )

        if self.step_per_collect is None:
            self.step_per_collect = len(self.train_envs)

        if self.update_per_step is None:
            self.update_per_step = 1 / self.step_per_collect

        # spaces
        self.observation_space = self.train_envs.observation_space[0]
        self.action_space = self.train_envs.action_space[0]

        if isinstance(self.observation_space, GymTuple):
            self.state_shape = (osp.shape or osp.n for osp in self.observation_space)
        else:
            self.state_shape = self.observation_space.shape or self.observation_space.n

        self.action_shape = self.action_space.shape or self.action_space.n
        self.reward_threshold = self.train_envs.spec[0].reward_threshold

        # episode per test
        if episode_per_test is None:
            self.episode_per_test = len(self.test_envs)
        else:
            self.episode_per_test = episode_per_test

        self._setup_policy(**policy_kwargs)
        self._setup_replay_buffer(replay_buffer)
        self._setup_collectors(train_collector, test_collector,
            exploration_noise_train, exploration_noise_test)
        self._setup_logger(logdir, task_name, method_name)

    @abc.abstractmethod
    def _setup_policy(self, **kwargs):
        """
        Performs setup for the policy, creating the ``self.policy`` attribute
        that is going to be passed to the trainer; optionally also exposes
        other related attributes.
        """
        raise NotImplementedError()

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
        if not seed is None: self.test_envs.seed(seed)
        self._test_fn(self.epoch, self.env_step)
        self.policy.eval()
    
        self.test_collector.reset()
        return self.test_collector.collect(
            n_episode=self.episode_per_test, render=render, **kwargs
        )

    def _setup_envs(
        self, task_name, task, test_task,
        train_env_class, train_envs,
        test_env_class, test_envs
    ):
        if task is None:
            task = lambda: gym.make(task_name)

        if test_task is None:
            test_task = task

        if isinstance(train_envs, int):
            if train_env_class is None:
                train_env_class = DummyVectorEnv if len(train_envs) == 1 else SubprocVectorEnv

            self.train_envs = train_env_class(
                [task for _ in range(train_envs)]
            )
        elif isinstance(train_envs, BaseVectorEnv):
            self.train_envs = train_envs
        else:
            raise TypeError(f"train_envs: a BaseVectorEnv or an integer expected, got '{train_envs}'.")

        if isinstance(test_envs, int):
            if test_env_class is None:
                test_env_class = DummyVectorEnv if len(test_envs) == 1 else SubprocVectorEnv

            self.test_envs = test_env_class(
                [test_task for _ in range(test_envs)]
            )
        elif isinstance(test_envs, BaseVectorEnv):
            self.test_envs = test_envs
        else:
            raise TypeError(f"test_envs: a BaseVectorEnv or an integer expected, got '{test_envs}'.")

    def _setup_logger(self, logdir, task_name, method_name):
        self.log_path = os.path.join(logdir, task_name, method_name)
        writer = SummaryWriter(self.log_path)
        self.logger = BasicLogger(writer)

    def _setup_replay_buffer(self, replay_buffer):
        if isinstance(replay_buffer, Number):
            self.replay_buffer = VectorReplayBuffer(
                replay_buffer, len(self.train_envs))
        elif isinstance(replay_buffer, ReplayBuffer):
            self.replay_buffer = replay_buffer
        else:
            self.replay_buffer = replay_buffer(len(self.train_envs))

    def _setup_collectors(self,
        train_collector, test_collector,
        exploration_noise_train,
        exploration_noise_test
    ):
        # collectors
        if isinstance(train_collector, Collector):
            self.train_collector = train_collector
        else:
            if train_collector is None: train_collector = Collector
            self.train_collector = train_collector(
                policy=self.policy,
                env=self.train_envs,
                buffer=self.replay_buffer,
                exploration_noise=exploration_noise_train
            )

        if isinstance(test_collector, Collector):
            self.test_collector = test_collector
        else:
            if test_collector is None: test_collector = Collector
            self.test_collector = test_collector(
                policy=self.policy,
                env=self.test_envs,
                buffer=None,
                exploration_noise=exploration_noise_test
            )

    def _save_fn(self, policy):
        torch.save(self.policy.state_dict(), os.path.join(self.log_path, 'policy.pth'))

    def _stop_fn(self, mean_rewards):
        if callable(self.stop_criterion):
            return self.stop_criterion(mean_rewards, self.reward_threshold, self)
        elif self.stop_criterion:
            return mean_rewards >= self.reward_threshold
        else:
            return True

    def _train_fn(self, epoch, env_step):
        self.epoch = epoch
        self.env_step = env_step
        for callback in self.train_callbacks:
            callback(epoch, env_step)

    def _test_fn(self, epoch, env_step):
        for callback in self.test_callbacks:
            callback(epoch, env_step)

    def _apply_seed(self):
        if not self._seed is None:
            np.random.seed(self._seed)
            torch.manual_seed(self._seed)
            self.train_envs.seed(self._seed)
            self.test_envs.seed(self._seed)

    def _init(self):
        self._inialized = True
        self._apply_seed()

    def construct_rlnet(
        self, module, module_params, state_shape, action_shape, **kwargs
    ):
        if not module_params is None:
            kwargs = kwargs.copy()
            kwargs.update(module_params)
        
        if isinstance(module, torch.nn.Module):
            module = module.to(self._device)
        else:
            if module is None: module = RLNetwork
            module = module(
                state_shape, action_shape,
                device=self._device,
                **kwargs
            ).to(self._device)

        return module

    def construct_optim(self, optim, optim_params, model_params):
        if optim_params is None: optim_params = {}
        if isinstance(optim, Optimizer):
            pass
        else:
            if optim is None: optim = torch.optim.Adam
            optim = optim(
                model_params,
                **optim_params
            )

        return optim

class OffPolicyAgent(Agent):
    def __init__(self,
        batch_size: int = 128,
        prefill_steps: int = 0,
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

        self.batch_size = batch_size

        if prefill_steps is None:
            self.prefill_steps = self.batch_size * len(self.train_envs)
        else:
            self.prefill_steps = prefill_steps

    def _init(self):
        super()._init()
        # prefill the replay buffer
        if self.prefill_steps:
            self.train_collector.collect(n_step=self.prefill_steps)

    def train(self, **kwargs) -> Dict[str, Union[float, str]]:
        """Runs off-policy training. The keyword arguments (if any) are used
        to update the arguments passed to ``offpolicy_trainer``. Most notably,
        perhaps, you can override the:
            * ``max_epoch``;
            * ``batch_size``;
            * ``step_per_epoch``;
            * ``step_per_collect``;
            * ``episode_per_test``;
            * ``update_per_step``;
            * ``logger``.

        Returns:
            dict: See :func:`~tianshou.trainer.gather_info`.
        """        

        if not self._inialized:
            self._init()

        self.policy.train()
        params = dict(
            policy=self.policy,
            train_collector=self.train_collector,
            test_collector=self.test_collector,
            max_epoch=self.max_epoch,
            step_per_epoch=self.step_per_epoch,
            step_per_collect=self.step_per_collect,
            episode_per_test=self.episode_per_test,
            batch_size=self.batch_size,
            update_per_step=self.update_per_step,
            stop_fn=self._stop_fn,
            train_fn=self._train_fn,
            test_fn=self._test_fn,
            save_fn=self._save_fn,
            logger=self.logger
        )

        params.update(kwargs)
        return offpolicy_trainer(**params)

class AgentPreset:
    def __init__(
        self,
        agent_class: Type[Agent],
        default_params: dict
    ):
        """The class used to construct presets.

        Args:
            agent_class (Type[Agent]): The class of the agent.
            default_params ([type]): The default parameters associated with
                this preset.
        """
        self.default_params = default_params
        self.agent_class = agent_class

    def __call__(self, *args, **kwargs):
        """Creates an instance of the agent, intialized using the default
        arguments from this preset, updated using the keyword arguments
        passed through this call.
        """
        params = self.default_params.copy()
        params.update(kwargs)
        return self.agent_class(*args, **params)
