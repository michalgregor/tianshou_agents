import os
import abc
from re import L
import torch
from .utils import StateDictObject
from typing import Callable

CallbackType = Callable[[int, int, int, 'ComponentAgent'], None]

class Callback(StateDictObject):
    @abc.abstractmethod
    def __call__(self, epoch, env_step, gradient_step, agent):
        raise NotImplementedError()

class ScheduleCallback(Callback):
    def __init__(self, setter, schedule):
        super().__init__()
        self.setter = setter
        self.schedule = schedule
        self._state_objs.append('schedule')

    def __call__(self, epoch, env_step, gradient_step, agent):
        val = self.schedule(epoch, env_step)
        self.setter(val)

class SaveCallback(Callback):
    def __init__(self, log_path='saved_models', fname="best_agent.pth"):
        super().__init__()
        self.log_path = log_path
        self.fname = fname

    def __call__(self, epoch, env_step, gradient_step, agent):
        state_dict = agent.state_dict()
        torch.save(state_dict, os.path.join(self.log_path, self.fname))

class CheckpointCallback(SaveCallback):
    def __init__(self, log_path='saved_models', fname="last_agent.pth", interval=1, method="epoch"):
        super().__init__(log_path, fname=fname)
        self.interval = interval
        self.method = method

    def __call__(self, epoch, env_step, gradient_step, agent):
        make_checkpoint = False

        if self.method == "epoch":
            make_checkpoint = epoch % self.interval == 0
        elif self.method == "env_step":
            make_checkpoint = env_step % self.interval == 0
        elif self.method == "gradient_step":
            make_checkpoint = gradient_step % self.interval == 0
        else:
            raise ValueError(f"Unknown method '{self.method}'.")

        if make_checkpoint:
            super().__call__(epoch, env_step, gradient_step, agent)
