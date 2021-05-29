# from https://github.com/BorealisAI/pommerman-baseline/blob/master/random_agent.py

from pommerman.agents import BaseAgent
from pommerman.constants import Action


class StaticAgent(BaseAgent):
    """ Static agent"""

    def act(self, obs, action_space):
        return Action.Stop.value
