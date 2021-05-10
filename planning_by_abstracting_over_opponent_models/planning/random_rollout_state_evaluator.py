import numpy as np
import torch

from planning_by_abstracting_over_opponent_models.planning.state_evaluator import StateEvaluator


class RandomRolloutStateEvaluator(StateEvaluator):
    def __init__(self, nb_players, nb_actions, pw_alphas, depth=None, heuristic_func=None):
        self.nb_players = nb_players
        self.nb_actions = nb_actions
        self.pw_alphas = pw_alphas
        self.depth = depth if depth is not None else 10000
        self.heuristic_func = heuristic_func

    def evaluate(self, env):
        game_state = env.get_game_state()
        initial_state = env.get_observations()
        step = 0
        done = False
        while not done and step < self.depth:
            actions = np.random.randint(low=0, high=self.nb_actions, size=self.nb_players, dtype=np.uint8)
            state, rewards, done = env.step(actions)
            step += 1
        env.set_game_state(game_state)
        rewards = rewards if done else self.heuristic_func(initial_state, state)
        rewards = torch.as_tensor(rewards).float()
        action_probs = torch.full((self.nb_players, self.nb_actions), 1 / self.nb_actions)
        return rewards, action_probs, self.pw_alphas
