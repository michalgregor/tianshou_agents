from .agent import Agent
from typing import Type

class AgentPreset:
    def __init__(
        self,
        agent_class: Type[Agent],
        default_params: dict
    ):
        """The class used to construct presets.

        Args:
            agent_class (Type[Agent]): The class of the agent.
            default_params ([type]): The default parameters associated with
                this preset.

        To see more info about the parameters, consult the docstring of
        ``self.agent_class``.
        """
        self.default_params = default_params
        self.agent_class = agent_class

    def __call__(self, *args, **kwargs):
        """Creates an instance of the agent, intialized using the default
        arguments from this preset, updated using the keyword arguments
        passed through this call.
        """
        params = self.default_params.copy()
        params.update(kwargs)
        return self.agent_class(*args, **params)

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