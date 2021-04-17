import pommerman
import torch
import torch.nn.functional as F
from pommerman.agents import BaseAgent

from planning_by_abstracting_over_opponent_models.config import gpu


def get_observation(state):
    features = state["board"]
    obs = torch.from_numpy(features)
    obs = obs.float().unsqueeze(0).unsqueeze(0).to(gpu)
    return obs


class Agent(BaseAgent):

    def __init__(self, agent_model):
        super().__init__()
        self.agent_model = agent_model

    def act(self, obs, action_space):
        agent_policy, _, _, _, _ = self.estimate(obs)
        agent_prob = F.softmax(agent_policy, dim=-1)
        agent_prob = agent_prob.view(-1)
        agent_action = agent_prob.argmax().item()
        # agent_action = agent_prob.multinomial(num_samples=1).detach()
        # agent_action = agent_action.item()
        return agent_action

    def estimate(self, obs):
        obs = get_observation(obs)
        return self.agent_model(obs)
