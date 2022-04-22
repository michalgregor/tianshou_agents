
from .component import Component
from tianshou.utils.logger.base import LOG_DATA_TYPE, BaseLogger
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Union, Dict, Any, Callable, Tuple

import torch
import os

class BaseLoggerComponent(Component, BaseLogger):
    """
    The base of logger components. Logger components are responsible for
    constructing a logger and assigning it to self.logger.
    
    Logger components are also responsible for keeping track of env_step, epoch,
    gradient_step, and episode counters. It also maintains the states of
    these counters using the StateDictObject interface inhereted from Component.
    """
    def __init__(self,
        train_interval: int = 1000,
        test_interval: int = 1,
        update_interval: int = 1000,
        **kwargs
    ):
        super().__init__(
            train_interval=train_interval,
            test_interval=test_interval,
            update_interval=update_interval,
            **kwargs
        )

class LoggerComponent(BaseLoggerComponent):
    def __init__(self,
        agent: "ComponentAgent",
        device: Union[str, int, torch.device] = "cpu",
        seed: int = None,
        log_path: Optional[str] = None,
        log_dir: Optional[str] = None,
        task_name: Optional[str] = None,
        method_name: Optional[str] = None,
        config_arg: int = None,
        component_class: Any = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.env_step = 0
        self.epoch = 0
        self.gradient_step = 0
        self.episode = 0

        if task_name is None:
            task_name = getattr(agent.component_train_collector, 'task_name', None)

        if method_name is None:
            method_name = getattr(agent.component_policy, 'method_name', None)

        assert task_name is not None, "task_name is not specified."
        assert method_name is not None, "method_name is not specified."

        if component_class is None:
            component_class = TensorboardLogger
           
        self.log_path = None
        if log_dir is None:
            if isinstance(config_arg, str): log_dir = config_arg
            if log_dir is None: log_dir = "log"
        
        if isinstance(config_arg, BaseLogger):
            self._wrapped_logger = config_arg
        else:
            if log_path is None:
                self.log_path = os.path.join(log_dir, task_name, method_name)

            writer = SummaryWriter(self.log_path)
            self._wrapped_logger = component_class(writer)
     
        self._state_objs.extend([
            'epoch',
            'env_step',
            'gradient_step',
            'episode',
        ])

    @property
    def logger(self):
        return self

    def reset_progress_counters(self):
        self.env_step = 0
        self.epoch = 0
        self.gradient_step = 0
        self.episode = 0

    def __getattr__(self, name):
        return getattr(self._wrapped_logger, name)

    def __dir__(self):
        d = set(self._wrapped_logger.__dir__())
        d.update(set(super().__dir__()))
        return d

    def write(self, step_type: str, step: int, data: LOG_DATA_TYPE) -> None:
        return self._wrapped_logger.write(step_type, step, data)

    def log_train_data(self, collect_result: dict, step: int) -> None:
        self.env_step += collect_result["n/st"]
        self.episode += collect_result["n/ep"]
        return self._wrapped_logger.log_train_data(collect_result, step)

    def log_test_data(self, collect_result: dict, step: int) -> None:
        return self._wrapped_logger.log_test_data(collect_result, step)

    def log_update_data(self, update_result: dict, step: int) -> None:
        self.gradient_step += 1
        return self._wrapped_logger.log_update_data(update_result, step)

    def save_data(
        self,
        epoch: int,
        env_step: int,
        gradient_step: int,
        save_checkpoint_fn: Optional[Callable[[int, int, int], None]] = None,
    ) -> None:
        self.epoch += 1
        return self._wrapped_logger.save_data(epoch, env_step, gradient_step, save_checkpoint_fn)

    def restore_data(self) -> Tuple[int, int, int]:
        return (self.epoch, self.env_step, self.gradient_step)
