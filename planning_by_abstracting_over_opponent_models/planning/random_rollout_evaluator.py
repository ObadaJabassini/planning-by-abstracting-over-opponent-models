from copy import deepcopy

import numpy as np
import torch

from planning_by_abstracting_over_opponent_models.planning.state_evaluator import StateEvaluator


class RandomRolloutEvaluator(StateEvaluator):
    def __init__(self, nb_players, action_space_size):
        self.nb_players = nb_players
        self.nb_opponents = self.nb_players - 1
        self.action_space_size = action_space_size

    def evaluate(self, env, state):
        snapshot = deepcopy(env)
        done = False
        while not done:
            actions = np.random.randint(low=0, high=self.action_space_size, size=self.nb_players).tolist()
            state, rewards, done, _ = snapshot.step(actions)
        rewards = torch.as_tensor(rewards)
        action_probs = torch.full((self.nb_players, self.action_space_size), 1 / self.action_space_size)
        opponent_influence = torch.full((self.nb_opponents, ), 1 / self.nb_opponents)
        return rewards, action_probs, opponent_influence
