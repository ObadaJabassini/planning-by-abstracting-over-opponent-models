import torch.nn.functional as F
from pommerman.agents import BaseAgent

from planning_by_abstracting_over_opponent_models.env import get_observation


class Agent(BaseAgent):

    def __init__(self, agent_model, nb_opponents, max_steps, device, stochastic=False):
        super().__init__()
        self.agent_model = agent_model
        self.nb_opponents = nb_opponents
        self.max_steps = max_steps
        self.device = device
        self.stochastic = stochastic

    def act(self, obs, action_space):
        agent_policy, _, _, _, _ = self.estimate(obs)
        agent_prob = F.softmax(agent_policy, dim=-1)
        agent_prob = agent_prob.view(-1)
        agent_action = agent_prob.argmax() if not self.stochastic else agent_prob.multinomial(num_samples=1)
        agent_action = agent_action.item()
        return agent_action

    def estimate(self, obs):
        obs = get_observation(obs, self.nb_opponents, self.max_steps, self.device)
        return self.agent_model(obs)
