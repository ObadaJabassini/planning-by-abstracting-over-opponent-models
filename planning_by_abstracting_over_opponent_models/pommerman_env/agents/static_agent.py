# from https://github.com/BorealisAI/pommerman-baseline/blob/master/random_agent.py

from pommerman.constants import Action

from planning_by_abstracting_over_opponent_models.pommerman_env.agents.pommerman_agent import PommermanAgent


class StaticAgent(PommermanAgent):
    """ Static agent"""

    def reset_agent(self):
        pass

    def act(self, obs, action_space):
        return Action.Stop.value
