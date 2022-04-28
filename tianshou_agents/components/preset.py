from typing import Type, Optional
from ..agent import ComponentAgent
from ..utils import derive_conf
from ..utils.config_router import BaseConfigRouter, DefaultConfigRouter

class AgentPreset:
    def __init__(
        self,
        default_params: dict,
        agent_class: Type[ComponentAgent] = ComponentAgent,
        config_router: Optional[BaseConfigRouter] = None
    ):
        """The class used to construct presets.

        Args:
            default_params ([type]): The default parameters associated with
                this preset.
            agent_class (Type[ComponentAgent]): The class of the agent.

        To see more info about the parameters, consult the docstring of
        ``self.agent_class``.
        """
        self.config_router = config_router or DefaultConfigRouter()
        self._default_params = default_params
        self.agent_class = agent_class
        
    def __call__(self, task_name, **kwargs):
        """Creates an instance of the agent, intialized using the default
        arguments from this preset, updated using the keyword arguments
        passed through this call.
        """
        params = dict(self._default_params, **kwargs, task_name=task_name)
        params = self.config_router(**params)
        return self.agent_class(config_router=self.config_router, **params)

    def derive_conf(self, update_conf=None):
        params = self.config_router(**self._default_params)
        return derive_conf(params, update_conf)

class AgentPresetWrapper(AgentPreset):
    def __init__(self, preset):
        self._preset = preset

    def __getattr__(self, name):
        return getattr(self._preset, name)

    def __dir__(self):
        d = set(self._preset.__dir__())
        d.update(set(super().__dir__()))
        return d

    def __call__(self, *args, **kwargs):
        return self._preset(*args, **kwargs)
