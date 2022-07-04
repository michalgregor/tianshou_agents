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
    if isinstance(shape, Sequence):
        if depth >= max_depth:
            return False
        else:
            for s in shape:
                if not _validate_shape(s, depth + 1, max_depth):
                    return False
        return True

    elif isinstance(shape, int):
        return True
    else:
        return False

def is_valid_shape(shape):
    return _validate_shape(shape, 0)

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
