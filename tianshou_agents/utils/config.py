def derive_conf(base_conf, update_conf):
    """
    Derive a config dict form a base config (base_conf) by updating it with
    changes from another config (update_conf).
    """
    return dict(base_conf, **update_conf)

def construct_config_object(
    obj, obj_type, default_obj_constructor=None,
    obj_kwargs={}, none_as_default=False,
    auto_enabled=True, callable_enabled=True,
    default_arg_name='config_arg'
):
    if obj is None:
        if none_as_default:
            obj = default_obj_constructor(**obj_kwargs)
        # otherwise obj stays None
    elif auto_enabled and isinstance(obj, str) and obj == 'auto':
        obj = default_obj_constructor(**obj_kwargs)
    elif isinstance(obj, obj_type):
        pass
    elif isinstance(obj, dict):
        kwargs = obj.copy()

        if default_obj_constructor is None:
            obj = kwargs.pop("__type__")
        else:
            obj = kwargs.pop("__type__", default_obj_constructor)

        obj_kwargs = dict(obj_kwargs, **kwargs)
        obj = obj(**obj_kwargs)
    elif callable_enabled and callable(obj):
        obj = obj(**obj_kwargs)
    else: # support for all other config arguments, e.g. int, float,
          #str (other than 'auto'), etc.
        def_arg = {default_arg_name: obj}
        obj = default_obj_constructor(**def_arg, **obj_kwargs)

    return obj
