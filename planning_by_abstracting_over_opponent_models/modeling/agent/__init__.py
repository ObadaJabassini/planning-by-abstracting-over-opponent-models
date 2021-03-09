import pommerman
import torch.nn.functional as F

class Agent(pommerman.agents.BaseAgent):

    def __init__(self, agent_model):
        super().__init__()
        self.agent_model = agent_model

    def act(self, obs, action_space):
        agent_policy, _, _ = self.estimate(obs)
        agent_prob = F.softmax(agent_policy, dim=-1)
        agent_action = agent_prob.multinomial(num_samples=1).detach()
        agent_action = agent_action.numpy()
        return agent_action

    def estimate(self, obs):
        return self.agent_model(obs)