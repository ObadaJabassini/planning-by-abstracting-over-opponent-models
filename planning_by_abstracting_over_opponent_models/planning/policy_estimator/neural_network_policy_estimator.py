import torch
import torch.nn.functional as F

from planning_by_abstracting_over_opponent_models.config import cpu
from planning_by_abstracting_over_opponent_models.planning.policy_estimator import PolicyEstimator


class NeuralNetworkPolicyEstimator(PolicyEstimator):

    def __init__(self, agent_id, agent_model, nb_actions, agent_pw_c=None, agent_pw_alpha=None, threshold=1e-3):
        self.agent_id = agent_id
        self.agent_model = agent_model
        self.nb_actions = nb_actions
        self.agent_pw_c = agent_pw_c
        self.agent_pw_alpha = agent_pw_alpha
        self.threshold = threshold

    def estimate(self, env):
        state = env.get_observations()
        obs = env.get_features(state)
        obs = obs[self.agent_id]
        obs = obs.unsqueeze(0)
        agent_action_log, _, opponents_action_log, opponent_influence = self.agent_model(obs)
        action_probs = self.estimate_action_probabilities(agent_action_log, opponents_action_log)
        attentions = opponent_influence.view(-1).to(cpu).detach()
        attentions[attentions <= self.threshold] = 0
        pw_alphas = attentions.tolist().copy()
        pw_alphas.insert(0, self.agent_pw_alpha)
        pw_cs = (attentions * self.nb_actions).tolist()
        pw_cs.insert(0, self.agent_pw_c)
        return action_probs, pw_cs, pw_alphas

    def estimate_action_probabilities(self, agent_action_log, opponent_action_log):
        """
        Estimate action probabilities using the agent model
        :param agent_action_log:
        :param opponent_action_log:
        :return:
        """
        agent_action_probs = F.softmax(agent_action_log, dim=-1)
        agent_action_probs = agent_action_probs.view((1, -1))
        opponent_action_probs = F.softmax(opponent_action_log, dim=-1)
        opponent_action_probs = opponent_action_probs.view(-1, agent_action_probs.size(1))
        probs = torch.vstack((agent_action_probs, opponent_action_probs)).to(cpu).detach()
        return probs