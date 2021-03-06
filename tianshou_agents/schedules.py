import abc
import numpy as np
from .utils import StateDictObject

class Schedule(StateDictObject):
    def __init__(self, method='step'):
        """
        :param method: One of 'step' / 'env_step, 'gradient_step' and 'epoch'.
        """
        super().__init__()
        self._method = method

    @abc.abstractmethod
    def get_value(self, step):
        raise NotImplementedError()

    def __call__(self, agent):
        if callable(self._method):
            return self.get_value(self._method(agent))
        elif self._method == 'step' or self._method == 'env_step':
            return self.get_value(agent.env_step)
        elif self._method == 'epoch':
            return self.get_value(agent.epoch)
        elif self._method == 'gradient_step':
            return self.get_value(agent.gradient_step)
        elif self._method == 'episode':
            val = self.get_value(agent.episode)
            return val
        else:
            raise ValueError(f"Unknown method '{self._method}'.")

class ConstSchedule(Schedule):
    def __init__(self, value, method='step'):
        super().__init__(method)
        self.value = value

    def get_value(self, step):
        return self.value

class LinearSchedule(Schedule):
    def __init__(self, final_step, init_val,
        final_val, first_step=0, method='step'
    ):
        super().__init__(method)
        assert(first_step < final_step)
        self.init_val = init_val
        self.final_val = final_val
        self.first_step = first_step
        self.final_step = final_step
        self.grad = (self.final_val - self.init_val) / (self.final_step - self.first_step)

    def get_value(self, step):
        if step < self.first_step:
            return self.init_val
        elif step >= self.final_step:
            return self.final_val
        else:
            return self.init_val + self.grad*(step-self.first_step)

class ExponentialSchedule(Schedule):
    def __init__(self, final_step, init_val,
        final_val, first_step=0, method='step'
    ):
        super().__init__(method)
        assert(first_step < final_step)
        self.init_val = init_val
        self.final_val = final_val
        self.first_step = first_step
        self.final_step = final_step
        self.coeff = np.power(self.final_val / self.init_val, 1/(self.final_step-self.first_step))

    def get_value(self, step):
        if step < self.first_step:
            return self.init_val
        elif step >= self.final_step:
            return self.final_val
        else:
            return self.init_val * self.coeff**(step-self.first_step)
