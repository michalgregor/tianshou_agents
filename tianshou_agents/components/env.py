from tianshou.env import BaseVectorEnv, DummyVectorEnv, SubprocVectorEnv
import gym

def extract_shape(space):
    if space is None:
        shape = None
    if isinstance(space, gym.spaces.Tuple):
        shape = (osp.shape or osp.n for osp in space)
    else:
        shape = space.shape or space.n

    return shape

def setup_envs(task, env_class, envs):
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

    return envs
