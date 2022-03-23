from tianshou.utils.net.common import MLP
from typing import Any, Dict, List, Tuple, Union, Optional, Sequence, Callable
from numbers import Number
from torch import nn
import numpy as np
import torch

DataType = Union[np.ndarray, torch.Tensor]

class RLNetwork(nn.Module):
    def __init__(
        self,
        state_shape: Union[int, Tuple[int], List[Tuple[int]]],
        action_shape: Union[int, Sequence[int]] = 0,
        model: Optional[Union[nn.Module, Callable[..., nn.Module]]] = None,
        device: Union[str, int, torch.device] = "cpu",
        flatten: bool = True,
        stateful: bool = False,
        softmax: bool = False,
        actions_as_input: bool = False,
        num_atoms: int = 1,
        dueling_param: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None,
        **model_kwargs
    ) -> None:
        super().__init__()
        self._device = device
        self.softmax = softmax
        self.num_atoms = num_atoms
        self.stateful = stateful
        self.flatten = flatten

        self._verify_state_shape(state_shape)

        if self.flatten:
            input_dim = int(np.prod(state_shape))
        else:
            input_dim = state_shape

        action_dim = int(np.prod(action_shape)) * num_atoms       
        
        if actions_as_input:
            if self.flatten:
                input_dim += action_dim
            else:
                if isinstance(input_dim, list):
                    input_dim += [action_dim]
                else:
                    input_dim = [input_dim, action_dim]

        self.use_dueling = dueling_param is not None

        if self.use_dueling:
            output_dim = 0
        elif actions_as_input:
            output_dim = 1
        else:
            output_dim = action_dim

        if isinstance(model, nn.Module):
            self.model = model
        else:
            if model is None: model = MLP
            self.model = model(input_dim, output_dim, device=device, **model_kwargs).to(device)

        if output_dim > 0:
            self.output_dim = output_dim
        elif hasattr(self.model, 'output_dim'):
            self.output_dim = self.model.output_dim
        else:
            if isinstance(input_dim, list):
                dummy_input = [torch.zeros(5, ind, device=device) for ind in input_dim]
            else:
                dummy_input = torch.zeros(5, input_dim, device=device)

            shape = self.model(dummy_input).shape[1:]
            assert len(shape) == 1
            self.output_dim = shape[0]

        if self.output_dim == 0:
            raise ValueError("The output_dim of a model cannot be zero; make sure that you check for num_outputs==0 if you are using a custom architecture.")

        if self.use_dueling:  # dueling DQN
            q_kwargs, v_kwargs = dueling_param  # type: ignore
            q_output_dim, v_output_dim = 0, 0

            if not actions_as_input:
                q_output_dim, v_output_dim = action_dim, num_atoms

            q_kwargs: Dict[str, Any] = {
                **q_kwargs, "input_dim": self.output_dim,
                "output_dim": q_output_dim}
            v_kwargs: Dict[str, Any] = {
                **v_kwargs, "input_dim": self.output_dim,
                "output_dim": v_output_dim}
            self.Q, self.V = MLP(**q_kwargs), MLP(**v_kwargs)
            self.output_dim = self.Q.output_dim

    def _verify_state_shape(self, state_shape):
        wrong = False

        if isinstance(state_shape, list):
            for sh in state_shape:
                if isinstance(sh, tuple):
                    for s in sh:
                        if not isinstance(s, Number):
                            wrong = True
                            break
                    
                    if wrong: break
                elif isinstance(sh, Number):
                    pass
                else:
                    wrong = True
        elif isinstance(state_shape, Number):
            pass
        elif isinstance(state_shape, tuple):
            for s in state_shape:
                if not isinstance(s, Number):
                    wrong = True
                    break
        else:
            wrong = True

        if wrong:
            raise TypeError(f"Expected ``state_shape`` to be a number, a tuple or a list of tuples, got '{state_shape}'.")

    def forward(
        self,
        args: Union[Tuple[DataType], List[DataType], DataType], 
        state: Any = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        """Mapping: s -> flatten (inside MLP)-> logits."""
        if not isinstance(args, tuple) and not isinstance(args, list):
            args = [args]
        args = tuple(torch.as_tensor(s, device=self._device, dtype=torch.float32)
                        for s in args)

        if self.flatten:
            args = [torch.flatten(s, start_dim=1) for s in args]
            args = (torch.cat(args),)

        if self.stateful:
            logits, state = self.model(*args, state=state)
        else:
            logits = self.model(*args)

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
