import torch
import torch.nn.functional as F

from planning_by_abstracting_over_opponent_models.config import cpu
from planning_by_abstracting_over_opponent_models.planning.state_evaluator import StateEvaluator


class NeuralNetworkStateEvaluator(StateEvaluator):
    def __init__(self, agent_id, nb_actions, agent_model, agent_pw_c, agent_pw_alpha=1, threshold=1e-3):
        self.agent_id = agent_id
        self.nb_actions = nb_actions
        self.agent_model = agent_model
        self.agent_pw_c = agent_pw_c
        self.agent_pw_alpha = agent_pw_alpha
        self.threshold = threshold

    def evaluate(self, env):
        state = env.get_observations()
        obs = env.get_features(state)
        obs = obs[self.agent_id]
        obs = obs.unsqueeze(0)
        agent_action_log, agent_value, opponents_action_log, opponent_values, opponent_influence = self.agent_model(obs)
        value_estimate = self.estimate_values(agent_value, opponent_values)
        action_probs_estimate = self.estimate_action_probabilities(agent_action_log, opponents_action_log)
        attentions = opponent_influence.view(-1).to(cpu).detach()
        attentions[attentions <= self.threshold] = 0
        pw_alphas = attentions.tolist().copy()
        pw_alphas.insert(0, self.agent_pw_alpha)
        pw_cs = attentions * self.nb_actions
        pw_cs = pw_cs.tolist()
        pw_cs.insert(0, self.agent_pw_c)
        return value_estimate, action_probs_estimate, pw_cs, pw_alphas

    def estimate_values(self, agent_value, opponent_values):
        """
        Estimate state's value using the agent model
        :param agent_value:
        :param opponent_values:
        :return:
        """
        agent_value = agent_value.view(-1).to(cpu)
        opponent_values = opponent_values.view(-1).to(cpu)
        value_estimate = torch.cat((agent_value, opponent_values)).to(cpu).detach()
        return value_estimate

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
