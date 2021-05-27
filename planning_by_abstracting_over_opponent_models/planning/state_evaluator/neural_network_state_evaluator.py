import torch
import torch.nn.functional as F

from planning_by_abstracting_over_opponent_models.learning.config import cpu
from planning_by_abstracting_over_opponent_models.planning.state_evaluator import StateEvaluator


class NeuralNetworkStateEvaluator(StateEvaluator):
    def __init__(self, agent_id, agent_model, agent_pw_alpha=1):
        self.agent_id = agent_id
        self.agent_model = agent_model
        self.agent_pw_alpha = agent_pw_alpha

    def evaluate(self, env):
        state = env.get_observations()
        obs = env.get_features(state)
        obs = obs[self.agent_id]
        obs = obs.unsqueeze(0)
        agent_action_log, agent_value, opponents_action_log, opponent_values, opponent_influence = self.agent_model(obs)
        value_estimate = self.estimate_values(agent_value, opponent_values)
        action_probs_estimate = self.estimate_action_probabilities(agent_action_log, opponents_action_log)
        pw_alphas = opponent_influence.view(-1).to(cpu).tolist()
        pw_alphas.insert(0, self.agent_pw_alpha)
        return value_estimate, action_probs_estimate, pw_alphas

    def estimate_values(self, agent_value, opponent_values):
        """
        Estimate state's value using the agent model
        :param agent_value:
        :param opponent_values:
        :return:
        """
        agent_value = agent_value.view(-1).to(cpu)
        opponent_values = opponent_values.view(-1).to(cpu)
        value_estimate = torch.cat((agent_value, opponent_values)).to(cpu)
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
        action_space_size = agent_action_probs.size(1)
        opponent_action_probs = F.softmax(opponent_action_log, dim=-1)
        opponent_action_probs = opponent_action_probs.view(-1, action_space_size)
        probs = torch.vstack((agent_action_probs, opponent_action_probs)).to(cpu)
        return probs
