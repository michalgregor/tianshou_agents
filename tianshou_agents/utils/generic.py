from functools import reduce

class NoDefaultProvided(object):
    pass

def getattrd(obj, name, default=NoDefaultProvided):
    """
    Same as getattr(), but allows dot notation lookup
    Discussed in:
    http://stackoverflow.com/questions/11975781
    """

    try:
        return reduce(getattr, name.split("."), obj)
    except AttributeError:
        if default != NoDefaultProvided:
            return default
        raise

def setattrd(obj, name, value):
    name_parts = name.split(".")

    if len(name_parts) > 1:
        obj = getattrd(obj, ".".join(name_parts[:-1]))
    
    setattr(obj, name_parts[-1], value)