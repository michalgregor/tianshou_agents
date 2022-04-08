import gym
import numpy as np
from tianshou.env import BaseVectorEnv
from typing import Any, List, Optional, Union, Tuple
from typing import Optional, Tuple, Union, Any

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
