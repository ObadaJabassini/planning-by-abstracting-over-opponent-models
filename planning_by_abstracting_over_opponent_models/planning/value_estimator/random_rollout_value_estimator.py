import numpy as np
import torch

from planning_by_abstracting_over_opponent_models.planning.value_estimator import ValueEstimator


class RandomRolloutValueEstimator(ValueEstimator):

    def __init__(self, nb_players, nb_actions):
        self.nb_players = nb_players
        self.nb_actions = nb_actions

    def estimate(self, env):
        game_state = env.get_game_state()
        done = False
        while not done:
            actions = np.random.randint(low=0, high=self.nb_actions, size=self.nb_players, dtype=np.uint8)
            state, rewards, done = env.step(actions)
        env.set_game_state(game_state)
        rewards = torch.as_tensor(rewards).float()
        return rewards


