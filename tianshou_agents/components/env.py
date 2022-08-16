from tianshou.env import BaseVectorEnv, DummyVectorEnv, SubprocVectorEnv
from tianshou.data import Batch
from typing import Union, Dict, Any, Sequence
from functools import singledispatch
import collections
import numpy as np
import torch
import gym

# extract the shape of the supplied observation space
@singledispatch
def extract_shape(space: gym.Space) -> Union[tuple, gym.Space]:
    """Extracts the shape of a gym.Space object.
    * For Tuple spaces, this returns a tuple of the shapes of the components.
    * For Dict spaces, this returns a dict of the shapes of the components.
    * For Discrete spaces, this returns the number of the space's elements.
    * For MultiDiscrete spaces, this returns a tuple containing the numbers
      of the each of the Discrete space's elements.
    * For other spaces with a shape attribute (such as Box spaces and
      MultiBinary spaces), this returns the shape, unless the shape is None.
    * For spaces that are not supported directly, the space object is returned
      as is.

    Support for new space types can be added by registering a custom function
    for the space type in the single-dispatch table.
    
    Args:
        space (gym.Space): The space object.

    Returns:
        Union[tuple, gym.Space]: The shape of the space or the space itself
            if the particular type of space is not supported by the
            extract_shape function.
    """
    if hasattr(space, 'shape') and not space.shape is None:
        if len(space.shape) == 1:
            return space.shape[0]
        else:
            return space.shape
    else:
      return space

@extract_shape.register(gym.spaces.Discrete)
def _extract_shape_discrete(space: gym.spaces.Discrete) -> int:
    return space.n

@extract_shape.register(gym.spaces.Tuple)
def _extract_shape_tuple(space: gym.spaces.Tuple) -> tuple:
    return tuple(extract_shape(s) for s in space.spaces)

@extract_shape.register(gym.spaces.Dict)
def _extract_shape_dict(space: gym.spaces.Dict) -> dict:
    return {k: extract_shape(s) for k, s in space.spaces.items()}

@extract_shape.register(gym.spaces.MultiDiscrete)
def _extract_shape_multidiscrete(space: gym.spaces.MultiDiscrete) -> tuple:
    return tuple(space.nvec[i] for i in range(len(space.nvec)))

# construct dummy space
@singledispatch
def construct_space(shape: Any) -> gym.Space:
    """Builds a dummy gym space, given a particular shape.

    Supports integers, sequences of integers and spaces nested using tuples
    and dicts. Given an integer or integer sequence, a Box space is constructed.
    """
    if shape is None:
        return None
    else:
        raise NotImplementedError('unsupported type: {}'.format(type(shape)))

@construct_space.register(collections.abc.Sequence)
def _construct_space_tuple(shape: Sequence) -> gym.spaces.Tuple:
    if not len(shape):
        return tuple()
    
    if isinstance(shape[0], int):
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)
    else:
        return gym.spaces.Tuple(map(construct_space, shape))

@construct_space.register(dict)
def _construct_space_dict(shape: Dict) -> gym.spaces.Dict:
    return gym.spaces.Dict({k: construct_space(v) for k, v in shape.items()})

@construct_space.register(int)
def _construct_space_int(shape: int) -> gym.spaces.Box:
    return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(shape,), dtype=np.float32)

# batch flattening
def validate_bound(ndims, bound):
    if bound < -ndims or bound > ndims-1:
        raise IndexError(f"Dimension out of range (expected to be in range of [{-ndims}, {ndims-1}], but got {bound})")

    return ndims + bound if bound < 0 else bound

def generic_flatten(
    x: Union[np.ndarray, torch.Tensor],
    start_dim=0, end_dim=-1
) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(x, np.ndarray):
        lbound = validate_bound(len(x.shape), start_dim)
        ubound = validate_bound(len(x.shape), end_dim)
        shape = x.shape[:lbound] + (-1,) + x.shape[ubound+1:]
        return x.reshape(shape)
    elif isinstance(x, torch.Tensor):
        return torch.flatten(x, start_dim=start_dim, end_dim=end_dim)
    else:
        raise ValueError('unsupported type: {}'.format(type(x)))

def generic_dummy_enc(labels: Union[np.ndarray, torch.Tensor], num_classes) -> Union[np.ndarray, torch.Tensor]:
    assert len(labels.shape) == 1 or (len(labels.shape) == 2 and labels.shape[1] == 1), \
        f"Expected shape of (batch_size,) or (batch_size, 1), but got {labels.shape}"
    
    if isinstance(labels, np.ndarray):
        return np.eye(num_classes)[labels.reshape(-1)]
    elif isinstance(labels, torch.Tensor):
        return torch.eye(num_classes)[labels.reshape(-1)]
    else:
        raise ValueError('unsupported type: {}'.format(type(labels)))

def is_pytorch(x: Union[Sequence, np.ndarray, torch.Tensor], op=all) -> bool:
    """Determines whether x is made purely out of PyTorch tensors."""
    if isinstance(x, np.ndarray):
        return False
    elif isinstance(x, torch.Tensor):
        return True
    elif isinstance(x, Sequence):
        return op(is_pytorch(item) for item in x)
    else:
        return False

@singledispatch
def batch_flatten(space: gym.Space, obs: Union[Batch, np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    if hasattr(space, 'shape') and not space.shape is None:
        return generic_flatten(obs, start_dim=1)
    else:
        raise NotImplementedError(f"Unknown space: `{space}`")

@batch_flatten.register(gym.spaces.Discrete)
def _batch_flatten_discrete(space: gym.spaces.Discrete, obs: Union[Batch, np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    return generic_dummy_enc(obs, space.n)

@batch_flatten.register(gym.spaces.Tuple)
def _batch_flatten_tuple(space: gym.spaces.Tuple, obs: Union[Batch, np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    args = [batch_flatten(space.spaces[i], np.stack(obs.transpose()[i]))
        for i in range(len(space.spaces))]

    if is_pytorch(obs):
        return torch.hstack(args)
    else:
        return np.hstack(args)

@batch_flatten.register(gym.spaces.Dict)
def _batch_flatten_dict(space: gym.spaces.Dict, obs: Union[Batch, np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    args = [batch_flatten(space.spaces[key], obs[key].squeeze(0)) for key in space.spaces]

    if is_pytorch(obs):
        return torch.hstack(args)
    else:
        return np.hstack(args)

@batch_flatten.register(gym.spaces.MultiDiscrete)
def _batch_flatten_discrete(space: gym.spaces.MultiDiscrete, obs: Union[Batch, np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    dummies = [
        generic_dummy_enc(obs[:, i], space.nvec[i])
            for i in range(len(space.nvec))
    ]

    if is_pytorch(obs):
        return torch.hstack(dummies)
    else:
        return np.hstack(dummies)

# batch2tensor
def to_tensor(x: Union[np.ndarray, torch.Tensor], **kwargs: Any) -> torch.Tensor:
    if not 'dtype' in kwargs:
        kwargs = {**kwargs, 'dtype': torch.float32}
        
    return torch.as_tensor(x, **kwargs)

@singledispatch
def batch2tensor(space: gym.Space, obs: Union[Batch, np.ndarray, torch.Tensor], **kwargs: Any) -> Union[dict, tuple, torch.Tensor]:
    """
    Converts a batch of observations to a tensor, transforming observation the
    same way as batch_flatten except for the final flattening step. Along the
    way, all arrays are converted to PyTorch tensors.

    Args:
        space: The gym space from which the observations are drawn.
        obs: The batch of observations to convert.
        kwargs: Additional keyword arguments to pass when constructing the
            tensor using as_tensor; dtype defaults to torch.float32.
    """
    if hasattr(space, 'shape') and not space.shape is None:
        return to_tensor(obs, **kwargs)
    else:
        raise NotImplementedError(f"Unknown space: `{space}`")

@batch2tensor.register(gym.spaces.Discrete)
def _batch2tensor_discrete(space: gym.spaces.Discrete, obs: Union[np.ndarray, torch.Tensor], **kwargs: Any) -> torch.Tensor:
    return to_tensor(generic_dummy_enc(obs, space.n), **kwargs)

@batch2tensor.register(gym.spaces.MultiDiscrete)
def _batch2tensor_discrete(space: gym.spaces.MultiDiscrete, obs: Union[np.ndarray, torch.Tensor], **kwargs: Any) -> tuple:
    return tuple(
        to_tensor(
            generic_dummy_enc(obs[:, i], space.nvec[i]),
            **kwargs
        ) for i in range(len(space.nvec))
    )

@batch2tensor.register(gym.spaces.Tuple)
def _batch2tensor_tuple(space: gym.spaces.Tuple, obs: Sequence, **kwargs: Any) -> tuple:
    obs_empty = np.empty((len(obs), len(space.spaces)), dtype=object)
    obs_empty[:] = obs
    
    return tuple(
        batch2tensor(space.spaces[i], np.stack(obs_empty.transpose()[i]), **kwargs) for i in range(len(space.spaces))
    )

@batch2tensor.register(gym.spaces.Dict)
def _batch2tensor_dict(space: gym.spaces.Dict, obs: Union[Batch, dict], **kwargs: Any) -> dict:
    return {key: batch2tensor(space.spaces[key], obs[key].squeeze(0), **kwargs) for key in space.spaces}

# shape validation
def _validate_shape(shape, depth, max_depth=2):
    real_depth = depth

    if isinstance(shape, collections.abc.Sequence):
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

def is_pure_shape(shape, return_depth=False, max_depth=2):
    """Returns whether shape represents a pure tensor shape. A pure shape
    is one of:
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

# env setup
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
