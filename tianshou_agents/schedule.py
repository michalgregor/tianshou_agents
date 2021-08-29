import abc
import numpy as np

class Schedule:
    def __init__(self, method='step'):
        """
        :param method: One of 'step' and 'epoch'.
        """
        self._method = method

    @abc.abstractmethod
    def get_value(self, step):
        raise NotImplementedError()

    def __call__(self, epoch, env_step):
        if self._method == 'step':
            return self.get_value(env_step)
        elif self._method == 'epoch':
            return self.get_value(epoch)
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
