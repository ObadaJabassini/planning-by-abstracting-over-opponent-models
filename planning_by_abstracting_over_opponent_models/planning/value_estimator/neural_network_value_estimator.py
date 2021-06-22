import torch

from planning_by_abstracting_over_opponent_models.config import cpu
from planning_by_abstracting_over_opponent_models.planning.value_estimator import ValueEstimator


class NeuralNetworkValueEstimator(ValueEstimator):

    def __init__(self, agent_id, agent_model):
        self.agent_id = agent_id
        self.agent_model = agent_model

    def estimate(self, env):
        state = env.get_observations()
        obs = env.get_features(state)
        obs = obs[self.agent_id]
        obs = obs.unsqueeze(0)
        _, agent_value, _, opponent_values, _ = self.agent_model(obs)
        value_estimate = self.estimate_values(agent_value, opponent_values)
        return value_estimate

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


