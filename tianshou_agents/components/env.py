from tianshou.env import BaseVectorEnv, DummyVectorEnv, SubprocVectorEnv
from collections.abc import Sequence
import gym

def extract_shape(space):
    if space is None:
        shape = None
    if isinstance(space, gym.spaces.Tuple):
        shape = (osp.shape or osp.n for osp in space)
    else:
        shape = space.shape or space.n

    return shape

def _validate_shape(shape, depth, max_depth=2):
    real_depth = depth

    if isinstance(shape, Sequence):
        if depth >= max_depth:
            return False, real_depth
        else:
            for s in shape:
                valid, tmp_depth = _validate_shape(s, depth + 1, max_depth)
                if tmp_depth > real_depth: real_depth = tmp_depth

                if not valid:
                    return False, real_depth

        return True, real_depth

    elif isinstance(shape, int):
        return True, real_depth
    else:
        return False, real_depth

def is_valid_shape(shape, return_depth=False, max_depth=2):
    """Returns whether shape represents a valid shape. A valid shape is one of:
        - an integer;
        - a sequence of integers;
        - a sequence containing integers or sequences of integers;
    i.e. there must be no more than 2 levels of nesting.

    Args:
        shape (_type_): _description_
        return_depth (bool, optional): Whether to also return the depth of the
            nested shape. Defaults to False.
        max_depth (int, optional): The maximum depth of the nested shape.
            Defaults to 2.       

    Returns:
        bool: whether shape is valid.

        or 

        Tuple[bool, int]: a (valid, depth) tuple if return_depth is True;
            valid is a boolean indicating whether shape is valid, and depth
            is an integer indicating the depth of the shape. If shape is
            invalid, depth evaluation terminates as soon as the invalidity
            is detected.
    """
    valid, depth = _validate_shape(shape, 0, max_depth=max_depth)
    return (valid, depth) if return_depth else valid

def setup_envs(task, env_class, envs, seed=None):
    if isinstance(envs, int):
        if env_class is None:
            env_class = DummyVectorEnv if envs == 1 else SubprocVectorEnv

        envs = env_class(
            [task for _ in range(envs)]
        )

    elif isinstance(envs, list):
        if env_class is None:
            env_class = DummyVectorEnv if len(envs) == 1 else SubprocVectorEnv

        envs = env_class([lambda: env if isinstance(env, gym.Env) else env for env in envs])
    elif isinstance(envs, BaseVectorEnv):
        pass
    elif isinstance(envs, gym.Env) and not hasattr(envs, "__len__"):
        envs = DummyVectorEnv([lambda: envs])
    else:
        raise TypeError(f"envs: a BaseVectorEnv or an integer expected, got '{envs}'.")

    if not seed is None:
        envs.seed(seed)

    return envs
