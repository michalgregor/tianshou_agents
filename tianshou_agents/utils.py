import gym
import numpy as np
from tianshou import env
from tianshou.env import BaseVectorEnv
from typing import Any, List, Optional, Union, Tuple
from collections import OrderedDict
from functools import reduce

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
