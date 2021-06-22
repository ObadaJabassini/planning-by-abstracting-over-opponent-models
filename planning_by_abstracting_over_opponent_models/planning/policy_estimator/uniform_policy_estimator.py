import torch

from planning_by_abstracting_over_opponent_models.planning.policy_estimator import PolicyEstimator


class UniformPolicyEstimator(PolicyEstimator):

    def __init__(self, nb_players, nb_actions, pw_cs, pw_alphas):
        self.nb_players = nb_players
        self.nb_actions = nb_actions
        self.pw_cs = pw_cs
        self.pw_alphas = pw_alphas

    def estimate(self, env):
        action_probs = torch.full((self.nb_players, self.nb_actions), 1 / self.nb_actions)
        return action_probs, self.pw_cs, self.pw_alphas
