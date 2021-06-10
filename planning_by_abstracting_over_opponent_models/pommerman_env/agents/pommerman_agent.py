import abc
from abc import ABC

from pommerman.agents import BaseAgent


class PommermanAgent(BaseAgent, ABC):
    @abc.abstractmethod
    def reset_agent(self):
        raise NotImplementedError()
