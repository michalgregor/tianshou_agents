from typing import Optional, Union, Callable, Dict, Any, List, Tuple
from copy import deepcopy
import warnings

def derive_conf(base_conf, update_conf=None):
    """
    Derive a config dict form a base config (base_conf) by updating it with
    changes from another config (update_conf).
    """
    config = deepcopy(base_conf)
    if not update_conf is None:
        config.update(update_conf)
    return config

class ConfigBuilder:
    """A class used to construct objects from configs.

    A config can be one of the following:
        - None: This is interpreted as an empty, default config if
            none_as_default is True. If none_as_default is False, the config
            will return None back in this case when contructing the config.
        - An instance of obj_type: This is intepreted as a pre-constructed
            instance. It will simply be returned back by the config builder.
        - A callable: This is interpreted as a callable that returns an
            instance of obj_type. The callable must be able to accept any 
            keyword arguments that are passed to the config builder as
            default_kwargs.
        - A dict: This is interpreted as a dict with keyword arguments to
            the object constructor. There are 3 special keyword arguments:
                - __inst__: This is interpreted as a pre-constructed
                    instance. It will simply be returned back by the config
                    builder. If the config dict contains any other keyword
                    arguments, they will be ignored; if verbose is True,
                    a warning will be issued to this effect.
                - __type__: This is interpreted as a callable that returns
                    an instance of obj_type. The callable must be able to
                    accept any regular keyword arguments that are part of
                    the config dict or that are passed to the config builder
                    as default_kwargs.
                - __arg__: This is a keyword argument that is used to pass
                    some kind of default argument to the object constructor.
                    It is passed as a keyword argument with the keyword being
                    equal to default_arg_name.
        - Any other type: This will be treated as a default argument to be
            passed to the default object constructor (equivalent to __arg__
            in a dict config). The keyword argument will be named
            according to default_arg_name.

    Args:
        obj_type (Any): The type of object to construct. Used to check if the
            config is a pre-constructed instance.
        default_obj_constructor (Callable[..., Any], optional): The default
            constructor to use if the config does not provide one.
        none_as_default (bool, optional): If True, the default constructor will
            be used if the config is None.
        callable_enabled (bool, optional): If True, a config that is a callable
            will be used as a constructor.
        default_arg_name (str, optional): If the config is not a dict, an
            instance, or a callable (unless callable_enabled is False), it
            is interpreted as an argument to the default constructor;
            default_arg_name is the name of the argument.
        verbose (bool, optional): If True, warnings will be printed in some
            edge cases.
    """

    def __init__(self,
        obj_type: Any,
        default_obj_constructor: Optional[Callable[..., Any]] = None,
        none_as_default: bool = False,
        callable_enabled: bool = True,
        default_arg_name: str = 'config_arg',
        verbose:bool = True
    ):
        self.obj_type = obj_type
        self.default_obj_constructor = default_obj_constructor
        self.none_as_default = none_as_default
        self.callable_enabled = callable_enabled
        self.default_arg_name = default_arg_name
        self.verbose = verbose

    def to_dict_config(self, config, copy=True):
        if isinstance(config, dict):
            if copy: config = config.copy()
        elif config is None:
            if self.none_as_default: config = {}
            # otherwise config stays None
        elif isinstance(config, self.obj_type):
            config = {'__inst__': config}
        elif self.callable_enabled and callable(config):
            config = {'__type__': config}
        else:
            config = {'__arg__': config}

        return config

    def __call__(self,
        config: Union[Any, Callable[..., Any], Dict[str, Any]],
        default_kwargs: dict = {}
    ):
        # transform config to a dict config; this gives us our own copy
        config = self.to_dict_config(config)

        # if config is None, just return None;
        # none_as_default is already handled by to_dict_config
        if config is None: return config

        # extract special arguments
        arg = config.pop('__arg__', None)
        inst = config.pop('__inst__', None)
        type = config.pop('__type__', None)

        if not arg is None: config[self.default_arg_name] = arg
        constructor = type if not type is None else self.default_obj_constructor

        if not inst is None:
            if self.verbose and len(config):
                warnings.warn(
                    "Config contains both a constructed instance '__inst__' " 
                    f"and extra arguments: {config}. These will be ignored."
                )

            return inst
        else:
            kwargs = dict(default_kwargs, **config)
            return constructor(**kwargs)
