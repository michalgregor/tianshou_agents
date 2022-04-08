from collections import OrderedDict
from .generic import getattrd, setattrd

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