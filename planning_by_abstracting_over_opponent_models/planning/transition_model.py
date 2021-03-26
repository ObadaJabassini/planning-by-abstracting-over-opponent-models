import gym
import torch
import torch.nn.functional as F
from planning_by_abstracting_over_opponent_models.utils import get_observation, cpu


class TransitionModel:
    def __init__(self, env: gym.Env, agent_model):
        self.env = env
        self.agent_model = agent_model

    def __call__(self, state, actions):
        # should modify, this will not work
        self.env.state = state
        state, rewards, done, _ = self.env.step(actions)
        return state, not done, rewards

    def probability(self, state):
        board = get_observation(state)
        agent_policy, _, opponent_policies, _ = self.agent_model(board)
        agent_prob = F.softmax(agent_policy, dim=-1)
        agent_prob = agent_prob.squeeze(0).to(cpu)
        agent_prob = agent_prob.reshape((1, -1))
        opponent_probs = F.softmax(opponent_policies, dim=-1)
        opponent_probs = opponent_probs.squeeze(0).to(cpu)
        probs = torch.vstack((agent_prob, opponent_probs))
        return probs
