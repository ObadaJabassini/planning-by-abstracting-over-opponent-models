from planning_by_abstracting_over_opponent_models.utils import get_board, cpu


class RewardModel:
    def __init__(self, agent_model):
        self.agent_model = agent_model

    def __call__(self, state):
        board = get_board(state)
        _, agent_value, _ = self.agent_model(board)
        agent_value = agent_value.to(cpu).item()
        # assume it is a zero-sum game
        return [agent_value, -agent_value]
