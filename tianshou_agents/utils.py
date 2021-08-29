import gym
import numpy as np
from tianshou.env import BaseVectorEnv
from tianshou.utils.log_tools import BasicLogger, WRITE_TYPE
from torch.utils.tensorboard import SummaryWriter
from typing import Any, List, Optional, Union, Tuple

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

class AgentLogger(BasicLogger):
    def __init__(self,
        agent: 'Agent',
        writer: Optional[SummaryWriter] = None,
        train_interval: int = 1000,
        test_interval: int = 1,
        update_interval: int = 1000,
        save_interval: int = 1,
    ):
        self.agent = agent
        
        super().__init__(
            writer=writer,
            train_interval=train_interval,
            test_interval=test_interval,
            update_interval=update_interval,
            save_interval=save_interval
        )

    def write(self, key: str, x: int, y: WRITE_TYPE, **kwargs: Any) -> None:
        if not self.writer is None:
            return super().write(key=key, x=x, y=y, **kwargs)

    def log_update_data(self, update_result: dict, step: int) -> None:
        self.agent.gradient_step = step
        return super().log_update_data(update_result=update_result, step=step)

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
            return super().restore_data()
