from .component import Component
from ..callbacks import CallbackType, SaveCallback, CheckpointCallback
from ..utils import construct_config_object
from typing import List, Optional, Callable, Union, Dict, Any, Tuple
from tianshou.trainer import BaseTrainer
from functools import partial
from numbers import Number
import torch
# for incremental trainer
from tianshou.trainer.utils import gather_info
from tianshou.utils import tqdm_config
from collections import deque
import tqdm

class TrainingFinishedDummyTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        pass

    def reset(self) -> None:
        pass

    def __next__(self) -> Union[None, Tuple[int, Dict[str, Any], Dict[str, Any]]]:
        return

    def test_step(self) -> Tuple[Dict[str, Any], bool]:
        pass

    def train_step(self) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
        pass

    def log_update_data(self, data: Dict[str, Any], losses: Dict[str, Any]) -> None:
        pass

    def policy_update_fn(self, data: Dict[str, Any], result: Dict[str, Any]) -> None:
        pass

    def run(self) -> Dict[str, Union[float, str]]:
        return {}

class TrainerComponent(Component):
    """
    
    Args:
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
    """
    
    def __init__(self,
        agent: 'ComponentAgent',
        device: Union[str, int, torch.device] = "cpu",
        seed: int = None,
        max_epoch: Optional[int] = None,
        step_per_epoch: Optional[int] = None,
        update_per_collect: int = 1,
        prefill_steps: int = 0,
        resume_from_log: bool = True,
        batch_size: int = 128,
        reward_threshold: Optional[float] = None,
        episode_per_test: Optional[int] = None,
        test_in_train: bool = False,
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
        config_arg: Optional[int] = None,
        component_class: Any = None,
        **trainer_kwargs
    ):
        super().__init__()
        self.max_epoch = max_epoch
        self.step_per_epoch = step_per_epoch
        self.update_per_collect = update_per_collect
        self.prefill_steps = prefill_steps
        self.batch_size = batch_size
        self.resume_from_log = resume_from_log
        self.reward_threshold = reward_threshold
        self.component_class = component_class
        self.trainer_kwargs = trainer_kwargs
        self.episode_per_test = episode_per_test
        self.test_in_train = test_in_train
        self._setup_args = {
            "train_callbacks": train_callbacks,
            "test_callbacks": test_callbacks,
            "save_best_callbacks": save_best_callbacks,
            "save_checkpoint_callbacks": save_checkpoint_callbacks,
        }
        self.init_done = False

        if isinstance(config_arg, int):
            if self.max_epoch is None:
                self.max_epoch = config_arg
        elif not config_arg is None:
            raise ValueError("config_arg must be an int or None")
     
        self.train_callbacks = []
        self.test_callbacks = []
        self.save_best_callbacks = []
        self.save_checkpoint_callbacks = []
        self.stop_criterion = stop_criterion

        self._state_objs.extend([
            'train_callbacks',
            'test_callbacks',
            'save_best_callbacks',
            'save_checkpoint_callbacks',
            'stop_criterion',
            'init_done',
        ])

    def setup(
        self,
        agent: 'ComponentAgent',
        device: Optional[Union[str, int, torch.device]] = None,
        seed: Optional[int] = None
    ):
        # train callbacks
        train_callbacks = construct_config_object(
            self._setup_args.pop('train_callbacks'), list,
            default_obj_constructor=lambda *args, **kwargs: [],
            obj_kwargs=dict(
                agent=agent,
                device=device,
                seed=seed,
            )
        )

        if train_callbacks is None:
            pass # no callback
        elif isinstance(train_callbacks, list):
            self.train_callbacks.extend(train_callbacks) # list of callbacks
        else:
            self.train_callbacks.append(train_callbacks) # single callback

        # test callbacks
        test_callbacks = construct_config_object(
            self._setup_args.pop('test_callbacks'), list,
            default_obj_constructor=lambda *args, **kwargs: [],
            obj_kwargs=dict(
                agent=agent,
                device=device,
                seed=seed,
            )
        )

        if test_callbacks is None:
            pass # no callback
        elif isinstance(test_callbacks, list):
            self.test_callbacks.extend(test_callbacks) # list of callbacks
        else:
            self.test_callbacks.append(test_callbacks) # single callback

        # try to get a default log_path from the logger
        if agent.logger is not None and agent.logger.log_path is not None:
            log_path = agent.logger.log_path
        else:
            log_path = None
            
        # save best callbacks
        obj_kwargs = {} if log_path is None else dict(log_path=log_path)

        save_best_callbacks = construct_config_object(
            self._setup_args.pop('save_best_callbacks'), list,
            default_obj_constructor=SaveCallback,
            default_arg_name="log_path",
            obj_kwargs=obj_kwargs
        )

        if save_best_callbacks is None:
            pass # no callback
        elif isinstance(save_best_callbacks, list):
            self.save_best_callbacks.extend(save_best_callbacks) # list of callbacks
        else:
            self.save_best_callbacks.append(save_best_callbacks) # single callback

        # save checkpoint callbacks
        obj_kwargs = {} if log_path is None else dict(log_path=log_path)
        obj_kwargs['interval'] = (max(int(self.max_epoch / 10), 1)
            if not self.max_epoch is None else 1)

        save_checkpoint_callbacks = construct_config_object(
            self._setup_args.pop('save_checkpoint_callbacks'), list,
            default_obj_constructor=CheckpointCallback,
            default_arg_name="log_path",
            obj_kwargs=obj_kwargs
        )

        if save_checkpoint_callbacks is None:
            pass # no callback
        elif isinstance(save_checkpoint_callbacks, list):
            self.save_checkpoint_callbacks.extend(save_checkpoint_callbacks) # list of callbacks
        else:
            self.save_checkpoint_callbacks.append(save_checkpoint_callbacks) # single callback

    def _init(self, agent):
        """
        Initializes the trainer.
        """

        if self.init_done: return 
        
        # prefill the replay buffer
        if self.prefill_steps and not agent.train_collector is None:
            agent.train_collector.collect(n_step=self.prefill_steps, random=True)

        self.init_done = True

    def make_trainer(self, agent, prefill_steps=None, **kwargs) -> BaseTrainer:
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
        # run init if not done yet
        self._init(agent)

        params = dict(
            policy=agent.policy,
            train_collector=agent.train_collector,
            test_collector=agent.test_collector,
            max_epoch=self.max_epoch,
            step_per_epoch=self.step_per_epoch,
            update_per_collect=self.update_per_collect,
            train_fn=partial(self._train_fn, agent),
            test_fn=partial(self._test_fn, agent),
            save_best_fn=partial(self._save_best_fn, agent),
            save_checkpoint_fn=partial(self._save_checkpoint_fn, agent),
            resume_from_log=self.resume_from_log,
            logger=agent.logger,
            component_class=self.component_class,
            reward_threshold=self.reward_threshold,
            episode_per_test=self.episode_per_test,
            batch_size=self.batch_size,
            test_in_train=self.test_in_train
        )

        params.update(self.trainer_kwargs)
        params.update(kwargs)

        # if no training is to be done (epoch >= max_epoch), create a dummy
        # trainer that does nothing; a regular trainer would still run at least
        # a single epoch
        if params["resume_from_log"]:
            # if the trainer's epoch counter resumes
            # from the global counter in the logger
            if agent.epoch >= params["max_epoch"]:
                return TrainingFinishedDummyTrainer()
        else:
            # if the trainer maintains its own separate counter
            if params["max_epoch"] == 0:
                return TrainingFinishedDummyTrainer()

        # autocompute update_per_step
        update_per_collect = params.pop("update_per_collect", None)
        update_per_step = params.pop("update_per_step", None)
        step_per_collect = params.get("step_per_collect", None)

        if update_per_step is None:
            if not update_per_collect is None and not step_per_collect is None:
                params["update_per_step"] = update_per_collect / step_per_collect

        # try to autocompute some params if not provided
        step_per_collect = params.get("step_per_collect", None)

        train_collector = params.get("train_collector", None)
        if (
            step_per_collect is None and
            not train_collector is None and
            not train_collector.env_num is None
        ):
            params["step_per_collect"] = train_collector.env_num

        train_envs = agent.train_envs
        if step_per_collect is None and not train_envs is None:
            params["step_per_collect"] = len(train_envs)
        
        episode_per_test = params.get("episode_per_test", None)
        test_envs = agent.test_envs
        if episode_per_test is None and not test_envs is None:
            params["episode_per_test"] = len(test_envs)

        # try to get the reward threshold from the env if not provided
        reward_threshold = params.pop("reward_threshold", None)
        if reward_threshold is None and not train_envs is None:
            reward_threshold = train_envs.spec[0].reward_threshold

        # if stop_fn not already overriden, set it up here
        if params.get("stop_fn", None) is None:
            params["stop_fn"] = partial(self._stop_fn, agent, reward_threshold)

        # assert that max_epoch is not None
        assert params["max_epoch"] is not None

        # the trainer increments max_epoch at the beginning of epoch and then
        # runs the check; i.e. we need to make max_epoch 4 if we actually want
        # to run 3 epochs
        params["max_epoch"] += 1

        # construct the trainer
        component_class = params.pop("component_class", None)
        return component_class(**params)

    # callbacks
    def _save_best_fn(self, agent, policy):
        for callback in self.save_best_callbacks:
            callback(agent.logger.epoch, agent.logger.env_step, agent.logger.gradient_step, agent)

    def _save_checkpoint_fn(self, agent, epoch, env_step, gradient_step):
        for callback in self.save_checkpoint_callbacks:
            callback(agent.logger.epoch, agent.logger.env_step, agent.logger.gradient_step, agent)

    def _stop_fn(self, agent, reward_threshold, mean_rewards):
        if callable(self.stop_criterion):
            return self.stop_criterion(mean_rewards, reward_threshold, agent)
        elif isinstance(self.stop_criterion, Number) and not isinstance(self.stop_criterion, bool):
            return mean_rewards >= self.stop_criterion
        elif (not reward_threshold is None and (
                (isinstance(self.stop_criterion, bool) and self.stop_criterion)
                or
                (isinstance(self.stop_criterion, str) and self.stop_criterion == 'auto')
            )
        ):
            return mean_rewards >= reward_threshold
        else:
            return False

    def _train_fn(self, agent, epoch, env_step):
        for callback in self.train_callbacks:
            callback(agent.epoch, agent.env_step, agent.logger.gradient_step, agent)

    def _test_fn(self, agent, epoch, env_step):
        for callback in self.test_callbacks:
            callback(agent.epoch, agent.env_step, agent.logger.gradient_step, agent)

class StepWiseTrainer:
    def __init__(self, trainer):
        self.trainer = trainer
        self._inner_gen = None

    def __getattr__(self, name):
        return getattr(self.trainer, name)

    def __dir__(self):
        d = set(self.trainer.__dir__())
        d.update(set(super().__dir__()))
        return d

    def __iter__(self):  # type: ignore
        self.trainer.reset()
        return self

    def _next_gen(self):
        """Perform one epoch (both train and eval)."""
        trainer = self.trainer

        trainer.epoch += 1
        trainer.iter_num += 1

        if trainer.iter_num > 1:

            # iterator exhaustion check
            if trainer.epoch >= trainer.max_epoch:
                return

            # exit flag 1, when stop_fn succeeds in train_step or test_step
            if trainer.stop_fn_flag:
                return

        # set policy in train mode
        trainer.policy.train()

        epoch_stat: Dict[str, Any] = dict()
        # perform n step_per_epoch
        with tqdm.tqdm(
            total=trainer.step_per_epoch, desc=f"Epoch #{trainer.epoch}", **tqdm_config
        ) as t:
            while t.n < t.total and not trainer.stop_fn_flag:
                data: Dict[str, Any] = dict()
                result: Dict[str, Any] = dict()
                if trainer.train_collector is not None:
                    data, result, trainer.stop_fn_flag = trainer.train_step()
                    t.update(result["n/st"])
                    if trainer.stop_fn_flag:
                        t.set_postfix(**data)
                        break
                else:
                    assert trainer.buffer, "No train_collector or buffer specified"
                    result["n/ep"] = len(trainer.buffer)
                    result["n/st"] = int(trainer.gradient_step)
                    t.update()

                trainer.policy_update_fn(data, result)
                t.set_postfix(**data)
                yield

            if t.n <= t.total and not trainer.stop_fn_flag:
                t.update()

        if not trainer.stop_fn_flag:
            trainer.logger.save_data(
                trainer.epoch, trainer.env_step, trainer.gradient_step, trainer.save_checkpoint_fn
            )
            # test
            if trainer.test_collector is not None:
                test_stat, trainer.stop_fn_flag = trainer.test_step()
                if not trainer.is_run:
                    epoch_stat.update(test_stat)

        if not trainer.is_run:
            epoch_stat.update({k: v.get() for k, v in trainer.stat.items()})
            epoch_stat["gradient_step"] = trainer.gradient_step
            epoch_stat.update(
                {
                    "env_step": trainer.env_step,
                    "rew": trainer.last_rew,
                    "len": int(trainer.last_len),
                    "n/ep": int(result["n/ep"]),
                    "n/st": int(result["n/st"]),
                }
            )
            info = gather_info(
                trainer.start_time, trainer.train_collector, trainer.test_collector,
                trainer.best_reward, trainer.best_reward_std
            )
            yield trainer.epoch, epoch_stat, info
        else:
            yield None

    def __next__(self) -> Union[None, Tuple[int, Dict[str, Any], Dict[str, Any]]]:
        if self._inner_gen is None:
            self._inner_gen = self._next_gen()

        try:
            ret = next(self._inner_gen)
        except StopIteration:
            self._inner_gen = self._next_gen()
            ret = next(self._inner_gen)

        return ret
    
    def run(self) -> Dict[str, Union[float, str]]:
        """Consume iterator.

        See itertools - recipes. Use functions that consume iterators at C speed
        (feed the entire iterator into a zero-length deque).
        """
        trainer = self.trainer

        try:
            trainer.is_run = True
            deque(self, maxlen=0)  # feed the entire iterator into a zero-length deque
            info = gather_info(
                trainer.start_time, trainer.train_collector, trainer.test_collector,
                trainer.best_reward, trainer.best_reward_std
            )
        finally:
            trainer.is_run = False

        return info
