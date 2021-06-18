import numpy as np
import torch

from planning_by_abstracting_over_opponent_models.planning.state_evaluator import StateEvaluator


class RandomRolloutStateEvaluator(StateEvaluator):
    def __init__(self, nb_players, nb_actions, pw_cs, pw_alphas):
        self.nb_players = nb_players
        self.nb_actions = nb_actions
        self.pw_alphas = pw_alphas
        self.pw_cs = pw_cs

    def evaluate(self, env):
        game_state = env.get_game_state()
        step = 0
        done = False
        while not done:
            actions = np.random.randint(low=0, high=self.nb_actions, size=self.nb_players, dtype=np.uint8)
            state, rewards, done = env.step(actions)
            step += 1
        env.set_game_state(game_state)
        rewards = torch.as_tensor(rewards).float()
        action_probs = torch.full((self.nb_players, self.nb_actions), 1 / self.nb_actions)
        return rewards, action_probs, self.pw_cs, self.pw_alphas
