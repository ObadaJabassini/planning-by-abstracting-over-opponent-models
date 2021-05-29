import random

from pommerman.agents import BaseAgent
from pommerman.constants import Action

from planning_by_abstracting_over_opponent_models.pommerman_env.agents import action_prune


class SmartRandomAgent(BaseAgent):
    """ random with filtered actions"""

    def act(self, obs, action_space):
        valid_actions = action_prune.get_filtered_actions(obs)
        if len(valid_actions) == 0:
            valid_actions.append(Action.Stop.value)
        return random.choice(valid_actions)


class SmartRandomAgentNoBomb(BaseAgent):
    """ random with filtered actions but no bomb"""

    def act(self, obs, action_space):
        valid_actions = action_prune.get_filtered_actions(obs)
        if Action.Bomb.value in valid_actions:
            valid_actions.remove(Action.Bomb.value)
        if len(valid_actions) == 0:
            valid_actions.append(Action.Stop.value)
        return random.choice(valid_actions)
