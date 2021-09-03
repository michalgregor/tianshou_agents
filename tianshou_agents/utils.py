import gym
import numpy as np
from tianshou import env
from tianshou.env import BaseVectorEnv
from tianshou.utils.logger.base import LOG_DATA_TYPE, BaseLogger
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter
from typing import Any, List, Optional, Union, Tuple, Callable

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

class AgentLoggerWrapper(BaseLogger):
    def __init__(self, agent: 'Agent', logger: BaseLogger):
        self.agent = agent
        self.logger = logger

    def __getattr__(self, name):
        return getattr(self.preset, name)

    def write(self, step_type: str, step: int, data: LOG_DATA_TYPE) -> None:
        return self.logger.write(step_type, step, data)

    def log_train_data(self, collect_result: dict, step: int) -> None:
        return self.logger.log_train_data(collect_result, step)

    def log_test_data(self, collect_result: dict, step: int) -> None:
        return self.logger.log_test_data(collect_result, step)

    def log_update_data(self, update_result: dict, step: int) -> None:
        self.agent.gradient_step = step
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
        epoch, env_step, gradient_step = (self.agent.epoch,
            self.agent.env_step, self.agent.gradient_step)

        if (
            not epoch is None and
            not env_step is None and
            not gradient_step is None
        ):
            return epoch, env_step, gradient_step
        else:
            return (0, 0, 0)
