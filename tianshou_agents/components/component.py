
from ..utils import StateDictObject
from abc import ABCMeta
from functools import partial
from typing import Union, Optional
import torch

class MetaComponent(ABCMeta):
    def __getitem__(cls, component_class):
        return partial(cls, component_class=component_class)

class Component(StateDictObject, metaclass=MetaComponent):
    """
    The base of all component objects.

    Each component is responsible for constructing a certain aspect of the
    reinforcement learning agent such a replay buffer, a collector, a policy,
    etc. The component is also responsible for storing the state of the
    component using the StateDictObject interface.
    
    Every component's constructor should take the following keyword arguments:
    * agent: The agent that the component is being constructed for.
    * device: The PyTorch device to use for the component.
    * seed: The random seed to use for the component.
    * config_arg: Components may optionally be constructed from a single
                  config argument, e.g. a replay buffer can be constructed
                  from a single int that specifies its capacity.

                  The user may also choose to provide a pre-constructed
                  component object (e.g. a pre-constructed collector,
                  replay buffer or policy) using this argument. Derived
                  classes should check whether config_arg is of the appropriate
                  type and make use of such pre-constructed objects.
    * component_class: Most components will construct objects of some specific
                       class by default, e.g. a VectorReplayBuffer.
                       However, this optional argument can be used to overload
                       that default class. The requirement is, of course, that
                       the specified component_class must have a constructor
                       that takes the same arguments as the default class.
    * **kwargs: Any other keyword arguments that should be passed to the
                component's constructor.

    A template constructor:
        def __init__(self,
            agent: 'ComponentAgent',
            device: Optional[Union[str, int, torch.device]] = None,
            seed: int = None,
            config_arg: int = None,
            component_class: Any = None,
            **kwargs
        )

    To derive a component type with a certain predefined value of
    component_class, you can also use the [] operator as a shorthand.
    E.g. to derive a collector component with component_class=AsyncCollector,
    you could write CollectorComponent[AsyncCollector] and pass
    this instead of CollectorComponent in any context where that is required.
    """

    def setup(
        self,
        agent: 'ComponentAgent',
        device: Optional[Union[str, int, torch.device]] = None,
        seed: Optional[int] = None
    ):
        """
        This method is called after all components are constructed. A component
        can use it e.g. to connect to other components – especially those that
        come after it in the construction sequence.
        """
        pass