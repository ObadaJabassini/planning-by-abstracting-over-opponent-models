import torch.nn.functional as F
from pommerman.agents import BaseAgent


class RLAgent(BaseAgent):

    def __init__(self, agent_id, agent_model, stochastic=False):
        super().__init__()
        self.agent_id = agent_id
        self.agent_model = agent_model
        self.stochastic = stochastic

    def act(self, obs, action_space):
        action_probs, _, _, _, _ = self.estimate(obs)
        action_probs = F.softmax(action_probs, dim=-1)
        action_probs = action_probs.view(-1)
        agent_action = action_probs.argmax() if not self.stochastic else action_probs.multinomial(num_samples=1)
        agent_action = agent_action.item()
        return agent_action

    def estimate(self, obs):
        obs = obs[self.agent_id]
        # (1, 18, 11, 11)
        obs = obs.unsqueeze(0)
        return self.agent_model(obs)
