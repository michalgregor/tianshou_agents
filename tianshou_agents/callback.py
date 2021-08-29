import abc
from .schedule import Schedule

class Callback:
    @abc.abstractmethod
    def __call__(self, epoch, env_step):
        raise NotImplementedError()

class ScheduleCallback(Callback):
    def __init__(self, setter, schedule):
        self.setter = setter
        self.schedule = schedule

    def __call__(self, epoch, env_step):
        val = self.schedule(epoch, env_step)
        self.setter(val)
