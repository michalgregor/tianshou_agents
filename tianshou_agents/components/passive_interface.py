import time
import torch
import warnings
import numpy as np
from typing import List, Optional, Callable, Union, Dict, Any, Tuple
from collections import deque
import tqdm
from .component import Component

from tianshou.data import (
    VectorReplayBuffer, ReplayBuffer, CachedReplayBuffer,
    ReplayBufferManager, Batch, to_numpy
)
from tianshou.policy import BasePolicy
from tianshou.trainer.utils import gather_info
from tianshou.utils import tqdm_config, DummyTqdm

class PassiveCollector:
    def __init__(
        self,
        policy: BasePolicy,
        buffer: Optional[ReplayBuffer] = None,
        preprocess_fn: Optional[Callable[..., Batch]] = None,
        num_envs: int = 1,
        exploration_noise: bool = False,
        validate_collect_args = True
    ) -> None:
        """
        preprocess_fn must not operate in-place; it can be called multiple times
        for the same observations.
        """
        super().__init__()

        self.validate_collect_args = validate_collect_args
        self._collect_args = {}
        self._collect_res = None
        self._col_gen = None

        self.env_num = num_envs
        self.exploration_noise = exploration_noise
        self.policy = policy
        self._assign_buffer(buffer)
        self.preprocess_fn = preprocess_fn
        self.reset()

        self.step_count = 0
        self.episode_count = 0
        self.episode_rews = None
        self.episode_lens = None
        self.episode_start_indices = None

    def _assign_buffer(self, buffer: Optional[ReplayBuffer]) -> None:
        """Check if the buffer matches the constraint."""
        if buffer is None:
            buffer = VectorReplayBuffer(self.env_num, self.env_num)
        elif isinstance(buffer, ReplayBufferManager):
            assert buffer.buffer_num >= self.env_num
            if isinstance(buffer, CachedReplayBuffer):
                assert buffer.cached_buffer_num >= self.env_num
        else:  # ReplayBuffer or PrioritizedReplayBuffer
            assert buffer.maxsize > 0
            if self.env_num > 1:
                if type(buffer) == ReplayBuffer:
                    buffer_type = "ReplayBuffer"
                    vector_type = "VectorReplayBuffer"
                else:
                    buffer_type = "PrioritizedReplayBuffer"
                    vector_type = "PrioritizedVectorReplayBuffer"
                raise TypeError(
                    f"Cannot use {buffer_type}(size={buffer.maxsize}, ...) to collect "
                    f"{self.env_num} envs,\n\tplease use {vector_type}(total_size="
                    f"{buffer.maxsize}, buffer_num={self.env_num}, ...) instead."
                )
        self.buffer = buffer

    def reset(self) -> None:
        """Reset all related variables in the collector."""
        self.data = None
        self.ready_env_ids = None
        self.reset_buffer()
        self.reset_stat()

    def reset_stat(self) -> None:
        """Reset the statistic variables."""
        self.collect_step, self.collect_episode, self.collect_time = 0, 0, 0.0

    def reset_buffer(self, keep_statistics: bool = False) -> None:
        """Reset the data buffer."""
        self.buffer.reset(keep_statistics=keep_statistics)

    def _empty_batch(self) -> Batch:
        """Return an empty batch."""
        return Batch(
            obs={}, act={}, rew={}, done={}, obs_next={}, info={}, policy={}
        )

    def _reset_state(self, data: Batch, id: Union[int, List[int]]) -> None:
        """Reset the hidden state: self.data.state[id]."""
        if hasattr(data.policy, "hidden_state"):
            state = data.policy.hidden_state  # it is a reference
            if isinstance(state, torch.Tensor):
                state[id].zero_()
            elif isinstance(state, np.ndarray):
                state[id] = None if state.dtype == object else 0
            elif isinstance(state, Batch):
                state.empty_(id)

    def _extract_if_batch(self, data, key, allow_other_keys=False):
        if isinstance(data, Batch):
            if not allow_other_keys and data.keys() != {key}:
                raise ValueError(f"If data is a Batch, it must only contain key '{key}', got '{data.keys()}'")
            data = data.__dict__[key]
        return data

    def compute_action(
        self,
        obs: Union[Batch, np.ndarray, torch.Tensor],
        done: Optional[Union[Batch, np.ndarray, torch.Tensor]],
        state: Optional[Batch],
        no_grad: bool = True,
        return_info: bool = False,
    ) -> Tuple[Batch, Batch]:
        obs = self._extract_if_batch(obs, "obs")
    
        # preprocess the observation
        if self.preprocess_fn:
            obs = self.preprocess_fn(
                obs=obs, env_id=np.arange(self.env_num)
            ).get("obs", obs)

        # reset states for the cases where episode ended
        if not done is None and not state is None:
            done = self._extract_if_batch(done, "done")
            for i in np.where(done)[0]:
                self._reset_state(state, i)
            
        data = self._empty_batch()
        if not state is None: data.update(**state)
        data.update(obs=obs, done=done)
        last_state = data.policy.get("hidden_state", None)

        if no_grad:
            with torch.no_grad():  # faster than retain_grad version
                # data.obs will be used by agent to get result
                result = self.policy(data, last_state)
        else:
            result = self.policy(data, last_state)
        
        # return the results in a batch
        policy = result.pop("policy", Batch())
        assert isinstance(policy, Batch)
        state = result.pop("state", None)
        if state is not None:
            policy.hidden_state = state  # save state into buffer

        # set up the state batch
        state = Batch(policy=policy, obs_preproc=data.obs)

        # apply exploration noise to the actions
        act = to_numpy(result.pop("act"))
        if self.exploration_noise:
            act = self.policy.exploration_noise(act, data)

        # remap the actions, store both raw and remapped in act_batch
        act_batch = Batch(
            act_raw=act,
            act=self.policy.map_action(act)
        )

        if return_info:
            return act_batch, state, result
        else:
            return act_batch, state

    def _observe_transition(
        self,
        data: Batch,
        env_ids: Optional[Union[np.ndarray, List[int]]] = None
    ) -> None:
        if not self.data is None:
            raise RuntimeError("The collector has not yet consumed the previous transition.")

        if not env_ids is None:
            self.ready_env_ids = np.asarray(env_ids)
        elif self.ready_env_ids is None:
            self.ready_env_ids = np.arange(self.env_num)           

        self.data = self._empty_batch()
        self.data.update(**data)

        # if obs has already been preprocessed, use it directly
        obs_preproc = self.data.pop('obs_preproc', None)

        # otherwise preprocess the observation
        if obs_preproc is None:
            obs_preproc = self.data.obs

            if self.preprocess_fn:
                obs_preproc = self.preprocess_fn(
                    obs=obs_preproc, env_id=self.ready_env_ids
                ).get("obs", obs_preproc)

        self.data.update(obs=obs_preproc)

        # make sure the action stored in the buffer is raw and not remapped
        act_raw = self.data.pop('act_raw', None)

        if act_raw is None:
            act_raw = self.policy.map_action_inverse(self.data.act)

        self.data.update(act=act_raw)

        # now preprocess the rest of the transition
        if self.preprocess_fn:
            self.data.update(
                self.preprocess_fn(
                    obs_next=self.data.obs_next,
                    rew=self.data.rew,
                    done=self.data.done,
                    info=self.data.info,
                    policy=self.data.policy,
                    env_id=self.ready_env_ids,
                )
            )
    
    def _make_collect(
        self,
        n_step: Optional[int] = None,
        n_episode: Optional[int] = None,
        random: bool = False,
        render: Optional[float] = None,
        no_grad: bool = True,
    ):
        self._collect_args = dict(
            n_step=n_step,
            n_episode=n_episode,
            random=random,
            render=render,
            no_grad=no_grad,
        )

        if n_step is not None:
            assert n_episode is None, (
                f"Only one of n_step or n_episode is allowed in Collector."
                f"collect, got n_step={n_step}, n_episode={n_episode}."
            )
            assert n_step > 0
            if not n_step % self.env_num == 0:
                warnings.warn(
                    f"n_step={n_step} is not a multiple of #env ({self.env_num}), "
                    "which may cause extra transitions collected into the buffer."
                )
            self.ready_env_ids = np.arange(self.env_num)
        elif n_episode is not None:
            assert n_episode > 0
            self.ready_env_ids = np.arange(min(self.env_num, n_episode))
        else:
            raise TypeError(
                "Please specify at least one (either n_step or n_episode) "
                "in AsyncCollector.collect()."
            )

        start_time = time.time()
        
        self.step_count = 0
        self.episode_count = 0
        self.episode_rews = []
        self.episode_lens = []
        self.episode_start_indices = []

        # once we are ready to receive transitions, we yield and wait for data
        self.data = None
        yield

        while True:
            if self.data is None:
                raise ValueError("observe_transition was not called since the last transition had been consumed.")
            
            assert len(self.data) == len(self.ready_env_ids)

            # keep done around for the next part
            done = self.data.done

            # add data into the buffer
            ptr, ep_rew, ep_len, ep_idx = self.buffer.add(
                self.data, buffer_ids=self.ready_env_ids
            )

            # set the transition as consumed
            self.data = None

            # collect statistics
            self.step_count += len(self.ready_env_ids)

            if np.any(done):
                env_ind_local = np.where(done)[0]
                self.episode_count += len(env_ind_local)
                self.episode_lens.append(ep_len[env_ind_local])
                self.episode_rews.append(ep_rew[env_ind_local])
                self.episode_start_indices.append(ep_idx[env_ind_local])

                # remove surplus env id from self.ready_env_ids
                # to avoid bias in selecting environments
                if n_episode:
                    surplus_env_num = len(self.ready_env_ids) - (n_episode - self.episode_count)
                    if surplus_env_num > 0:
                        mask = np.ones_like(self.ready_env_ids, dtype=bool)
                        mask[env_ind_local[:surplus_env_num]] = False
                        self.ready_env_ids = self.ready_env_ids[mask]

            if (n_step and self.step_count >= n_step) or \
                    (n_episode and self.episode_count >= n_episode):
                break

            # once we have consumed the observed transition, we yield
            yield

        # generate statistics
        self.collect_step += self.step_count
        self.collect_episode += self.episode_count
        self.collect_time += max(time.time() - start_time, 1e-9)

        if self.episode_count > 0:
            rews, lens, idxs = list(
                map(
                    np.concatenate,
                    [self.episode_rews, self.episode_lens, self.episode_start_indices]
                )
            )
            rew_mean, rew_std = rews.mean(), rews.std()
            len_mean, len_std = lens.mean(), lens.std()
        else:
            rews, lens, idxs = np.array([]), np.array([], int), np.array([], int)
            rew_mean = rew_std = len_mean = len_std = 0

        yield {
            "n/ep": self.episode_count,
            "n/st": self.step_count,
            "rews": rews,
            "lens": lens,
            "idxs": idxs,
            "rew": rew_mean,
            "len": len_mean,
            "rew_std": rew_std,
            "len_std": len_std,
        }

    def make_collect(
        self,
        n_step: Optional[int] = None,
        n_episode: Optional[int] = None,
        random: bool = False,
        render: Optional[float] = None,
        no_grad: bool = True,
    ):
        if not self._col_gen is None:
            raise RuntimeError("The collector is already collecting.")

        self._col_gen = self._make_collect(
            n_step=n_step,
            n_episode=n_episode,
            random=random,
            render=render,
            no_grad=no_grad,
        )

        next(self._col_gen)

    def observe_transition(
        self,
        data: Batch,
        env_ids: Optional[Union[np.ndarray, List[int]]] = None
    ) -> Union[None, Dict[str, Any]]:
        self._observe_transition(data, env_ids)
        ret = next(self._col_gen)

        if not ret is None:
            self._col_gen = None
            self._collect_res = ret

        return ret

    @property
    def is_collecting(self):
        return self._col_gen is not None

    def collect(
        self,
        n_step: Optional[int] = None,
        n_episode: Optional[int] = None,
        random: bool = False,
        render: Optional[float] = None,
        no_grad: bool = True,
    ) -> Dict[str, Any]:
        if self._collect_res is None:
            raise ValueError("make_collector has not yielded since collect was called last.")

        args = dict(
            n_step=n_step,
            n_episode=n_episode,
            random=random,
            render=render,
            no_grad=no_grad,
        )

        if self.validate_collect_args and self._collect_args != args:
            raise ValueError(
                f"PassiveCollector.collect() was called with different arguments than the previous make_collector. "
                f"Previous call: {self._collect_args}. "
                f"Current call: {args}. "
                "To disable this check, set PassiveCollector.validate_collect_args=False."
            )

        collect_res = self._collect_res
        self._collect_res = None
        return collect_res

class StepWiseTrainer:
    def __init__(self, trainer):
        self.trainer = trainer
        self._inner_gen = None
        self.iter_called = False

    def __getattr__(self, name):
        return getattr(self.trainer, name)

    def __dir__(self):
        d = set(self.trainer.__dir__())
        d.update(set(super().__dir__()))
        return d

    def __iter__(self):  # type: ignore
        self.iter_called = True
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
                return True

            # exit flag 1, when stop_fn succeeds in train_step or test_step
            if trainer.stop_fn_flag:
                return True

        # set policy in train mode
        trainer.policy.train()

        epoch_stat: Dict[str, Any] = dict()

        if self.show_progress:
            progress = tqdm.tqdm
        else:
            progress = DummyTqdm

        # perform n step_per_epoch
        with progress(
            total=trainer.step_per_epoch, desc=f"Epoch #{trainer.epoch}", **tqdm_config
        ) as t:
            while t.n < t.total and not trainer.stop_fn_flag:
                if t.n: yield
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
            return trainer.epoch, epoch_stat, info
        else:
            return None

    def __next__(self) -> Union[None, Tuple[int, Dict[str, Any], Dict[str, Any]]]:
        if not self.iter_called:
            iter(self)

        if self._inner_gen is None:
            self._inner_gen = self._next_gen()

        try:
            ret = next(self._inner_gen)
        except StopIteration as e:
            self._inner_gen = None
            ret = e.value

            if isinstance(ret, bool) and ret:
                raise StopIteration from e

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

class PassiveInterface(Component):
    """A component that initializes the passive training interface.

    The following keyword arguments can be passed to construct
    the passive collector:
        num_envs: int = None,
        policy: Optional[BasePolicy] = None,
        replay_buffer: Optional[ReplayBuffer] = None,
        preprocess_fn: Optional[Callable[..., Batch]] = None,
        exploration_noise: bool = False,

    If a 'trainer' argument is passed it is interpreted as a preconstructed
    instance of the trainer. Otherwise, the trainer is constructed using 
    make_trainer, which also consumes any remaining keyword arguments.
    """

    def __init__(self,
        agent: 'ComponentAgent',
        device: Optional[Union[str, int, torch.device]] = None,
        seed: Optional[int] = None,
        config_arg: Optional[PassiveCollector] = None,
        component_class: Any = None,
        **kwargs
    ):
        super().__init__()

        self._state_objs.extend([
            'collector'
        ])

        # resolve args
        policy = kwargs.pop("policy", None)
        if policy is None: policy = agent.policy

        replay_buffer = kwargs.pop("replay_buffer", None)
        if replay_buffer is None: replay_buffer = agent.buffer

        num_envs = kwargs.pop("num_envs", None)
        if num_envs is None and not replay_buffer is None:
            num_envs = getattr(replay_buffer, "buffer_num", 1)

        if "preprocess_fn" in kwargs:
            preprocess_fn = kwargs.pop("preprocess_fn")
        elif not agent.train_collector is None:
            preprocess_fn = agent.train_collector.preprocess_fn
        else:
            preprocess_fn = None

        if "exploration_noise" in kwargs:
            exploration_noise = kwargs.pop("exploration_noise")
        elif not agent.train_collector is None:
            exploration_noise = agent.train_collector.exploration_noise
        else:
            exploration_noise = False

        # create the passive collector if a pre-constructed one is not passed
        if not config_arg is None:
            if isinstance(config_arg, PassiveCollector):
                self.collector = config_arg
            else:
                raise ValueError(f"Invalid config_arg: {config_arg}; expected PassiveCollector")
        else:
            if component_class is None:
                component_class = PassiveCollector

            self.collector = component_class(
                policy=policy, buffer=replay_buffer, preprocess_fn=preprocess_fn,
                num_envs=num_envs, exploration_noise=exploration_noise,
            )

        # create the trainer if a pre-constructed one is not passed
        if "trainer" in kwargs:
            self.trainer = StepWiseTrainer(kwargs["trainer"])
        else:
            trainer_kwargs = dict(
                train_collector=self.collector,
                max_epoch=float('inf')
            )

            trainer_kwargs.update(kwargs)
            trainer_kwargs["policy"] = policy
            trainer_kwargs["buffer"] = replay_buffer

            self.trainer = StepWiseTrainer(agent.make_trainer(**trainer_kwargs))
