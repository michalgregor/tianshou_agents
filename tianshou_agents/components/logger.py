
from ..utils import StateDictObject
from tianshou.utils.logger.base import LOG_DATA_TYPE, BaseLogger
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Union, Dict, Any, Callable, Tuple

import torch
import os

class LoggerComponent(StateDictObject, BaseLogger):
    def __init__(self,
        logger: Optional[Union[str, Dict[str, Any], BaseLogger]],
        task_name: str, method_name: str,
        device: Union[str, int, torch.device] = "cpu",
        seed: int = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.env_step = 0
        self.epoch = 0
        self.gradient_step = 0
        self.episode = 0

        self.log_path = None
        if isinstance(logger, BaseLogger):
            pass
        elif logger is None:
            self.log_path = os.path.join("log", task_name, method_name)
            writer = SummaryWriter(self.log_path)
            logger = TensorboardLogger(writer)
        elif isinstance(logger, str):
            self.log_path = os.path.join(logger, task_name, method_name)
            writer = SummaryWriter(self.log_path)
            logger = TensorboardLogger(writer)
        else:
            logger_params = logger.copy()
            make_logger = logger_params.pop("__type__", TensorboardLogger)

            if make_logger == TensorboardLogger:
                writer = logger_params.get("writer")

                if writer is None:
                    log_path = logger_params.pop("log_path")
                    log_dir = logger_params.pop("log_dir", "log")

                    if not log_path is None:
                        self.log_path = log_path
                    else:
                        self.log_path = os.path.join(log_dir, task_name, method_name)

                    logger_params['writer'] = SummaryWriter(self.log_path)

                else:
                    self.log_path = writer.log_dir

            logger = make_logger(**logger_params)

        self.logger = logger

        self._state_objs.extend([
            'epoch',
            'env_step',
            'gradient_step',
            'episode',
        ])

    def reset_progress_counters(self):
        self.env_step = 0
        self.epoch = 0
        self.gradient_step = 0
        self.episode = 0

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
        self.env_step += collect_result["n/st"]
        self.episode += collect_result["n/ep"]
        return self.logger.log_train_data(collect_result, step)

    def log_test_data(self, collect_result: dict, step: int) -> None:
        return self.logger.log_test_data(collect_result, step)

    def log_update_data(self, update_result: dict, step: int) -> None:
        self.gradient_step += 1
        return self.logger.log_update_data(update_result, step)

    def save_data(
        self,
        epoch: int,
        env_step: int,
        gradient_step: int,
        save_checkpoint_fn: Optional[Callable[[int, int, int], None]] = None,
    ) -> None:
        self.epoch += 1
        return self.logger.save_data(epoch, env_step, gradient_step, save_checkpoint_fn)

    def restore_data(self) -> Tuple[int, int, int]:
        return (self.epoch, self.env_step, self.gradient_step)
