def derive_conf(base_conf, update_conf):
    """
    Derive a config dict form a base config (base_conf) by updating it with
    changes from another config (update_conf).
    """
    return dict(base_conf, **update_conf)

def construct_config_object(
    obj, obj_type, default_obj_constructor=None, obj_kwargs={}
):
    if obj is None:
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
    else:
        obj = obj(**obj_kwargs)

    return obj
