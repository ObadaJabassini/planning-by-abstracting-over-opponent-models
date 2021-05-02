from copy import deepcopy

import numpy as np
import torch

from planning_by_abstracting_over_opponent_models.planning.state_evaluator import StateEvaluator


class RandomRolloutEvaluator(StateEvaluator):
    def __init__(self, nb_players, nb_actions):
        self.nb_players = nb_players
        self.nb_opponents = self.nb_players - 1
        self.nb_actions = nb_actions

    def evaluate(self, env):
        env._init_game_state = env.get_json_info()
        done = False
        while not done:
            actions = np.random.randint(low=0, high=self.nb_actions, size=self.nb_players).tolist()
            state, rewards, done, _ = env.step(actions)
        env.reset()
        rewards = torch.as_tensor(rewards)
        action_probs = torch.full((self.nb_players, self.nb_actions), 1 / self.nb_actions)
        opponent_influence = torch.full((self.nb_opponents, ), 1 / self.nb_opponents)
        return rewards, action_probs, opponent_influence
