import gym
import numpy as np
from tianshou.env import BaseVectorEnv
from typing import Any, List, Optional, Union, Tuple
from collections import OrderedDict
from functools import reduce
from tianshou.utils.logger.base import LOG_DATA_TYPE, BaseLogger
from tianshou.policy import BasePolicy
from tianshou.data import Collector
from typing import Optional, Tuple, Callable, Union, Any

class NoDefaultProvided(object):
    pass

def getattrd(obj, name, default=NoDefaultProvided):
    """
    Same as getattr(), but allows dot notation lookup
    Discussed in:
    http://stackoverflow.com/questions/11975781
    """

    try:
        return reduce(getattr, name.split("."), obj)
    except AttributeError:
        if default != NoDefaultProvided:
            return default
        raise

def setattrd(obj, name, value):
    name_parts = name.split(".")

    if len(name_parts) > 1:
        obj = getattrd(obj, ".".join(name_parts[:-1]))
    
    setattr(obj, name_parts[-1], value)

class StateDictObject:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._state_objs = []

    def _make_state_dict(self, obj):
        if isinstance(obj, list):
            state_dict = [self._make_state_dict(o) for o in obj]
        elif hasattr(obj, 'state_dict'):
            state_dict = obj.state_dict()
        else:
            state_dict = obj

        return state_dict

    def _write_state_dict(self, obj, state_dict, setter):
        if isinstance(obj, list):
            if not isinstance(state_dict, list):
                raise TypeError("The object is a list, but the state_dict is not.")

            if len(obj) != len(state_dict):
                raise ValueError("The object and the state_dict are lists of different lengths.")

            for io in range(len(obj)):
                def setter(val):
                    obj[io] = val
                self._write_state_dict(obj[io], state_dict[io], setter)

        elif hasattr(obj, 'load_state_dict'):
            obj.load_state_dict(state_dict)
        else:
            setter(state_dict)

    def state_dict(self):
        state_dict = OrderedDict()

        for name in self._state_objs:
            obj = getattrd(self, name)
            state_dict[name] = self._make_state_dict(obj)

        return state_dict

    def load_state_dict(self, state_dict):
        for name in self._state_objs:
            def setter(val):
                setattrd(self, name, val)

            obj = getattrd(self, name)
            self._write_state_dict(obj, state_dict[name], setter)

class VectorEnvRenderWrapper(gym.Wrapper):
    def __init__(self, env: BaseVectorEnv):
        super().__init__(env)

    def __len__(self) -> int:
        """Return len(self), which is the number of environments."""
        return self.env.env_num

    def reset(self, id: Optional[Union[int, List[int], np.ndarray]] = None) -> np.ndarray:
        return self.env.reset(id)

    def step(
        self,
        action: np.ndarray,
        id: Optional[Union[int, List[int], np.ndarray]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.env.step(action, id)

    def seed(
        self, seed: Optional[Union[int, List[int]]] = None
    ) -> List[Optional[List[int]]]:
        return self.env.seed(seed)

    def render(self, **kwargs: Any) -> List[Any]:
        """Render all of the environments."""
        self.env._assert_is_not_closed()
        if self.env.is_async and len(self.env.waiting_id) > 0:
            raise RuntimeError(
                f"Environments {self.env.waiting_id} are still stepping, cannot "
                "render them now.")
        return self.env.workers[0].render(**kwargs)

    def close(self) -> None:
        return self.env.close()

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        return self.env.normalize_obs(obs)

class LoggerWrapper(BaseLogger):
    def __init__(self, logger: BaseLogger, **kwargs):
        super().__init__(**kwargs)
        self.logger = logger

    def __getattr__(self, name):
        return getattr(self.logger, name)

    def __dir__(self):
        d = set(self.logger.__dir__())
        d.update(set(super().__dir__()))
        return d

    @property
    def unwrapped(self):
        return self.logger

    def write(self, step_type: str, step: int, data: LOG_DATA_TYPE) -> None:
        return self.logger.write(step_type, step, data)

    def log_train_data(self, collect_result: dict, step: int) -> None:
        return self.logger.log_train_data(collect_result, step)

    def log_test_data(self, collect_result: dict, step: int) -> None:
        return self.logger.log_test_data(collect_result, step)

    def log_update_data(self, update_result: dict, step: int) -> None:
        self.gradient_step = step
        return self.logger.log_update_data(update_result, step)

    def save_data(
        self,
        epoch: int,
        env_step: int,
        gradient_step: int,
        save_checkpoint_fn: Optional[Callable[[int, int, int], None]] = None,
    ) -> None:
        return self.logger.save_data(epoch, env_step, gradient_step, save_checkpoint_fn)

    def restore_data(self) -> Tuple[int, int, int]:
        epoch, env_step, gradient_step = (self.epoch,
            self.env_step, self.gradient_step)

        if (
            not epoch is None and
            not env_step is None and
            not gradient_step is None
        ):
            return epoch, env_step, gradient_step
        else:
            return (0, 0, 0)

class PolicyWrapper(BasePolicy):
    def __init__(self, policy: BasePolicy, **kwargs):
        super().__init__(**kwargs)
        self.__dict__["policy"] = policy

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def __setattr__(self, name, value):
        if name in self.__dict__:
            self.__dict__[name] = value
        else:
            setattr(self.policy, name, value)

    def __dir__(self):
        d = set(self.policy.__dir__())
        d.update(set(super().__dir__()))
        return d

    @property
    def unwrapped(self):
        return self.policy

class Wrapper(Collector):
    def __init__(self, collector: Collector, **kwargs):
        super().__init__(**kwargs)
        self.__dict__["collector"] = collector

    def __getattr__(self, name):
        return getattr(self.collector, name)

    def __setattr__(self, name, value):
        if name in self.__dict__:
            self.__dict__[name] = value
        else:
            setattr(self.collector, name, value)

    def __dir__(self):
        d = set(self.collector.__dir__())
        d.update(set(super().__dir__()))
        return d

    @property
    def unwrapped(self):
        return self.collector