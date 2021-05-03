from copy import deepcopy

import numpy as np
import torch

from planning_by_abstracting_over_opponent_models.planning.state_evaluator import StateEvaluator


class RandomRolloutStateEvaluator(StateEvaluator):
    def __init__(self, nb_players, nb_actions, depth=None, heuristic_func=None):
        self.nb_players = nb_players
        self.nb_actions = nb_actions
        self.depth = depth if depth is not None else 10000
        self.heuristic_func = heuristic_func

    def evaluate(self, env):
        rewards = env._get_rewards()
        done = env._get_done()
        if not done:
            env._init_game_state = env.get_json_info()
            initial_state = env.get_observations()
            step = 0
            while not done and step < self.depth:
                actions = np.random.randint(low=0, high=self.nb_actions, size=self.nb_players).tolist()
                state, rewards, done, _ = env.step(actions)
                rewards = rewards[:self.nb_players]
                step += 1
            env.reset()
            rewards = rewards if done else self.heuristic_func(initial_state, state)
        rewards = torch.as_tensor(rewards).float()
        action_probs = torch.full((self.nb_players, self.nb_actions), 1 / self.nb_actions)
        nb_opponents = self.nb_players - 1
        opponent_influence = torch.full((nb_opponents, ), 1 / nb_opponents)
        return rewards, action_probs, opponent_influence
