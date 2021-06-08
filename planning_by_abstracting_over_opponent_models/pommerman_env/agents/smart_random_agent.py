import random

from pommerman.agents import BaseAgent
from pommerman.constants import Action

from planning_by_abstracting_over_opponent_models.pommerman_env.agents import action_prune
from planning_by_abstracting_over_opponent_models.pommerman_env.agents.pommerman_agent import PommermanAgent


class SmartRandomAgent(PommermanAgent):
    """ random with filtered actions"""

    def reset(self):
        self.last_obs = None
        self.last_last_obs = None

    def __init__(self, no_bomb=True):
        super().__init__()
        self.no_bomb = no_bomb
        self.last_obs = None
        self.last_last_obs = None

    def act(self, obs, action_space):
        valid_actions = action_prune.get_filtered_actions(obs)
        if self.no_bomb and Action.Bomb.value in valid_actions:
            valid_actions.remove(Action.Bomb.value)
        if len(valid_actions) == 0:
            valid_actions.append(Action.Stop.value)
        action = random.choice(valid_actions)
        self.last_last_obs = self.last_obs
        self.last_obs = obs
        return action
