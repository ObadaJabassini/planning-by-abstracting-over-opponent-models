import torch

from planning_by_abstracting_over_opponent_models.utils import get_observation, cpu


class RewardModel:
    def __init__(self, agent_model):
        self.agent_model = agent_model

    def __call__(self, state):
        board = get_observation(state)
        _, agent_value, _, opponent_values = self.agent_model(board)
        agent_value = agent_value.view(-1).to(cpu)
        opponent_values = opponent_values.view(-1).to(cpu)
        return torch.cat((agent_value, opponent_values))
