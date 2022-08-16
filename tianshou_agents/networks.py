import warnings
from tianshou.utils.net.common import MLP as _tia_MLP, ModuleType
from tianshou.utils.net.continuous import ActorProb
from .utils import ConfigBuilder
from .components.env import extract_shape, batch_flatten, construct_space, batch2tensor
from typing import Any, Dict, List, Tuple, Union, Optional, Sequence, Callable, Type, TypeVar
from numbers import Number
from torch import nn
import numpy as np
import warnings
import torch
import gym

T = TypeVar("T")
RLNetworkDataType = Union[np.ndarray, torch.Tensor]

class ActionTop(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()

        if output_shape:
            self.output_shape = output_shape
            self.top = nn.Linear(input_shape, output_shape)
        else:
            self.output_shape = input_shape
            self.top = nn.Identity()

    def forward(self, x):
        return self.top(x)

class MLP(_tia_MLP):
    """Simple MLP backbone.

    Create a MLP of size input space shape * hidden_sizes[0] * hidden_sizes[1] * ...
    * hidden_sizes[-1] * action space shape

    :param int input_shape: dimension of the input vector.
    :param int output_shape: dimension of the output vector. If set to 0, there
        is no final linear layer.
    :param hidden_sizes: shape of MLP passed in as a list, not including
        input_shape and output_shape.
    :param norm_layer: use which normalization before activation, e.g.,
        ``nn.LayerNorm`` and ``nn.BatchNorm1d``. Default to no normalization.
        You can also pass a list of normalization modules with the same length
        of hidden_sizes, to use different normalization module in different
        layers. Default to no normalization.
    :param activation: which activation to use after each layer, can be both
        the same activation for all layers if passed in nn.Module, or different
        activation for different Modules if passed in a list. Default to
        nn.ReLU.
    :param device: which device to create this model on. Default to None.
    :param linear_layer: use this module as linear layer. Default to nn.Linear.
    :param bool flatten_input: whether to flatten input data. Default to True.
    """

    def __init__(
        self,
        input_shape: Union[
            int,
            Sequence[int]
        ],
        output_shape: Union[
            int,
            Sequence[int]
         ] = None,
        hidden_sizes: Sequence[int] = (),
        norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
        activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU,
        device: Optional[Union[str, int, torch.device]] = None,
        linear_layer: Type[nn.Linear] = nn.Linear,
        flatten_input: bool = True,
        **kwargs
    ) -> None:
        if isinstance(input_shape, Number):
            input_shape = (input_shape,)

        if isinstance(output_shape, Number):
            output_shape = (output_shape,)

        if output_shape is None:
            output_shape = (0,)

        if flatten_input:
            input_dim = int(np.prod(input_shape))
            output_dim = int(np.prod(output_shape))
        else:
            input_dim = input_shape[-1]
            output_dim = output_shape[-1]

        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_sizes=hidden_sizes,
            norm_layer=norm_layer,
            activation=activation,
            device=device,
            linear_layer=linear_layer,
            flatten_input=flatten_input,
            **kwargs
        )

    @property
    def output_shape(self):
        return self.output_dim

class RLNetwork(nn.Module):
    """
    A convenience wrapper around nn.Modules that augments them with
    common interfaces useful for reinforcement learning.

    Args:
        observation_space (Union[gym.spaces.Space, Tuple[int]]):
            The space of observations that are going to be presented 
            to the network. This is used when constructing the model here
            from the provided spec. It is not going to be taken into account if
            a pre-constructed module is provided.

            Optionally, one may pass a shape instead – in that case a space
            will be constructed automatically using construct_space.

        action_space (gym.spaces.Space): The requisite action space (related
            to the shape of the network's output). Note that if this is set to
            None, that means the model itself needs to determine the size of
            its output because its last layers is going to be a hidden layer
            and some more layers are going to be stacked on top of it. For that
            reason, the output_shape passed to the model is also going to be
            None and the model needs to handle it accordingly, e.g. using the
            ActionTop convenience module:

                self.dense = nn.Linear(256, 128)
                self.relu = nn.ReLU()

                self.action_top = ActionTop(128, output_shape)
                self.output_shape = self.action_top.output_shape
            
            or by checking the output_shape manually along the lines of:

                if not self.output_shape is None:
                    self.dense2 = nn.Linear(128, output_shape)
                    self.relu2 = nn.ReLU()
                    self.output_shape = output_shape
                else:
                    self.output_shape = 128
            
        model (Union[
            nn.Module,
            Callable[..., nn.Module],
            Dict[str, Any]
        ], optional):
            The actual neural model to use as a ConfigBuilder spec. This
            is constructed using the class's model_builder attribute. The
            model can either be an nn.Module, in which case it is used
            directly or else it can be a callable or a ConfigBuilder dict.

            If constructed here, the following arguments are passed to the
            module's constructor:
                * input_shape: the shape of the input;
                * output_shape: the shape of the output (check action_shape
                    to see how output_shape=None is to be handled);
                * device: the torch device that the model should be using;
                * **model_kwargs: any other keyword arguments passed to
                  RLNetwork end up here;

        device (Union[str, int, torch.device]): The torch device to use.

        flatten (bool, optional): Specifies whether the observation shape
            should be flattened before being passed to the model. Since by
            default the model is an MLP, this is True by default. Switch
            it to False when flattening is not required, e.g. when the input
            it 2D and you are going to apply 2D convolution to it.

        stateful (bool): When set to True, the model is also passed a state
            when being called and returns an (output, state) pair as its
            output.

        softmax (bool): Whether to apply a softmax layer over the last
            layer's output.

        actions_as_input (bool): When True, actions are taken to be
            concatenated to the observation, i.e. they action_shape
            co-determines the input shape of the model and has no
            influence on its output shape.

        num_atoms (int): an order to expand to the net of distributional RL.
        Default to 1 (not use).

        dueling_param (Tuple[
            Dict[str, Any],
            Dict[str, Any]
        ], optional): whether to use dueling network to calculate Q values
            (for Dueling DQN). If you want to use dueling option, you need
            to pass a tuple of two dict objects (for Q and for V), e.g.:

            dueling_param=(
                {"hidden_sizes": [128, 128]}, # Q_param
                {"hidden_sizes": [128, 128]} # V_param
            )

            These are going to be used to set up two MLPs required for
            the dueling architecture.

        model_output_shape (Union[int, Tuple[int]], optional):
            The output shape of model. If not specified, the output shape
            is inferred from the action space, if possible, or taken from
            the model.output_shape attribute if it exists.

        extract_obs_shape (Callable[[gym.spaces.Space], Tuple[int, ...]], optional):
            A callable that takes an observation space and returns a tuple
            of ints representing the shape of the observation.

        **model_kwargs: Any other keyword arguments are passed to the
            model's constructor (unless the model is pre-constructed
            in which case they are ignored).
    """

    model_builder = ConfigBuilder(
        obj_type=nn.Module,
        default_obj_constructor=MLP,
        none_as_default=True
    )

    def __init__(
        self,
        observation_space: Union[gym.spaces.Space, Tuple[int]],
        action_space: Union[gym.spaces.Space, Tuple[int]] = None,
        model: Optional[Union[
            nn.Module,
            Callable[..., nn.Module],
            Dict[str, Any]
        ]] = None,
        device: Union[str, int, torch.device] = "cpu",
        flatten: bool = True,
        extract_tensors: bool = True,
        stateful: bool = False,
        softmax: bool = False,
        actions_as_input: bool = False,
        num_atoms: int = 1,
        dueling_param: Optional[Tuple[
            Dict[str, Any],
            Dict[str, Any]
        ]] = None,
        model_output_shape = None,
        extract_obs_shape: Callable[[gym.spaces.Space], Tuple[int, ...]] = None,
        **model_kwargs
    ) -> None:
        super().__init__()
        self._device = device
        self.softmax = softmax
        self.num_atoms = num_atoms
        self.stateful = stateful
        self.flatten = flatten
        self.extract_tensors = extract_tensors
        self.actions_as_input = actions_as_input

        if isinstance(observation_space, gym.spaces.Space):
            self.observation_space = observation_space
        else:
            self.observation_space = construct_space(observation_space)

        if isinstance(action_space, gym.spaces.Space):
            self.action_space = action_space
        else:
            self.action_space = construct_space(action_space)

        if extract_obs_shape is None:
            extract_obs_shape = extract_shape

        if self.flatten:
            input_shape = gym.spaces.flatdim(self.observation_space)
        else:
            input_shape = extract_obs_shape(self.observation_space)

        if self.action_space is None:
            action_dim = None
        else:
            action_dim = gym.spaces.flatdim(self.action_space) * self.num_atoms
        
        if self.actions_as_input and not action_dim is None:
            if self.flatten:
                input_shape += action_dim
            else:
                input_shape = [input_shape, action_dim]

        self.use_dueling = dueling_param is not None

        if self.use_dueling:
            output_shape = None
        elif self.actions_as_input:
            output_shape = 1
        else:
            output_shape = action_dim

        self.model = self.model_builder(
            model,
            default_kwargs=dict(model_kwargs,
                input_shape=input_shape,
                output_shape=output_shape,
                device=device
            ),
        ).to(device)

        if model_output_shape:
            output_shape = model_output_shape
        else:
            if not output_shape:
                output_shape = getattr(self.model, 'output_shape', None)

        if not output_shape:
            raise ValueError("The output_shape of a model cannot be zero "
                "or None; make sure that you check output_shape if you are "
                "using a custom architecture.")

        if self.use_dueling:  # dueling DQN
            if action_dim == 0:
                warnings.warn("Trying to use dueling, but action_dim is 0.")

            q_kwargs, v_kwargs = dueling_param  # type: ignore
            q_output_shape, v_output_shape = 0, 0

            if not self.actions_as_input:
                q_output_shape, v_output_shape = action_dim, num_atoms

            q_kwargs: Dict[str, Any] = {
                **q_kwargs, "input_shape": output_shape,
                "output_shape": q_output_shape}
            v_kwargs: Dict[str, Any] = {
                **v_kwargs, "input_shape": output_shape,
                "output_shape": v_output_shape}

            self.Q, self.V = MLP(device=device, **q_kwargs), MLP(device=device, **v_kwargs)
            self.Q = self.Q.to(device)
            self.V = self.V.to(device)

        if self.num_atoms > 1:
            self.output_shape = tuple(output_shape) + (self.num_atoms,)
        else:
            self.output_shape = output_shape

    @property
    def output_dim(self):
        return self.output_shape

    def forward(
        self,
        obs: Union[Sequence[RLNetworkDataType], RLNetworkDataType],
        act: Optional[Union[Sequence[RLNetworkDataType], RLNetworkDataType]] = None,
        state: Any = None,
        info: Dict[str, Any] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Any]:
        """Mapping: s -> flatten (inside MLP)-> logits."""

        if info is None:
            info = {}

        if not self.actions_as_input and not act is None:
            act = None
            warnings.warn(
                "The actions_as_input flag is False, but act is not None. "
                "Note that act will be ignored."
            )
            
        if self.flatten:
            inputs = batch_flatten(self.observation_space, obs)
            inputs = torch.as_tensor(inputs, device=self._device)
        else:
            inputs = batch2tensor(self.observation_space, obs, device=self._device)
            
        if not act is None:
            if self.flatten:
                flat_action = batch_flatten(self.action_space, act)
                flat_action = torch.as_tensor(flat_action, device=self._device)
                inputs = torch.cat((inputs, flat_action), dim=1)
                act = None
            else:
                act = batch2tensor(self.action_space, act, device=self._device)

        if act is None:
            args = (inputs,)
        else:
            args = (inputs, act)

        if self.stateful:
            logits, state = self.model(*args, state=state, **kwargs)
        else:
            logits = self.model(*args, **kwargs)

        batch_size = logits.shape[0]
        if self.use_dueling:  # Dueling DQN
            q, v = self.Q(logits), self.V(logits)
            if self.num_atoms > 1:
                q = q.view(batch_size, -1, self.num_atoms)
                v = v.view(batch_size, -1, self.num_atoms)
            logits = q - q.mean(dim=1, keepdim=True) + v
        elif self.num_atoms > 1:
            logits = logits.view(batch_size, -1, self.num_atoms)
        if self.softmax:
            logits = torch.softmax(logits, dim=-1)

        return logits, state

class Critic(nn.Module):
    """Simple critic network. Will create an actor operated in continuous \
    action space with structure of preprocess_net ---> 1(q value).

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param int preprocess_net_output_dim: the output dimension of
        preprocess_net.
    :param linear_layer: use this module as linear layer. Default to nn.Linear.
    :param bool flatten_input: whether to flatten input data for the last layer.
        Default to True.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    .. seealso::

        Please refer to :class:`~tianshou.utils.net.common.Net` as an instance
        of how preprocess_net is suggested to be defined.
    """

    def __init__(
        self,
        preprocess_net: nn.Module,
        hidden_sizes: Sequence[int] = (),
        device: Union[str, int, torch.device] = "cpu",
        preprocess_net_output_dim: Optional[int] = None,
        linear_layer: Type[nn.Linear] = nn.Linear,
        flatten_input: bool = True,
    ) -> None:
        super().__init__()
        self.device = device
        self.preprocess = preprocess_net
        self.output_dim = 1
        input_dim = getattr(preprocess_net, "output_dim", preprocess_net_output_dim)
        self.last = MLP(
            input_dim,  # type: ignore
            1,
            hidden_sizes,
            device=self.device,
            linear_layer=linear_layer,
            flatten_input=flatten_input,
        )

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        act: Optional[Union[np.ndarray, torch.Tensor]] = None,
        info: Dict[str, Any] = {},
    ) -> torch.Tensor:
        """Mapping: (s, a) -> logits -> Q(s, a)."""
        # note: casting to tensors is to be handled by preprocess_net
        logits, hidden = self.preprocess(obs=obs, act=act, info=info)
        logits = self.last(logits)
        return logits
