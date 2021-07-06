import torch.nn.functional as F

from planning_by_abstracting_over_opponent_models.pommerman_env.agents.pommerman_agent import PommermanAgent


class RLAgent(PommermanAgent):

    def reset_agent(self):
        pass

    def __init__(self, agent_id, agent_model, stochastic=False):
        super().__init__()
        self.agent_id = agent_id
        self.agent_model = agent_model
        self.stochastic = stochastic

    def act(self, obs, action_space):
        action_probs, _, _, _ = self.estimate(obs)
        action_probs = F.softmax(action_probs, dim=-1).view(-1)
        agent_action = action_probs.argmax() if not self.stochastic else action_probs.multinomial(num_samples=1)
        agent_action = agent_action.item()
        return agent_action

    def estimate(self, obs):
        obs = obs[self.agent_id]
        # (1, 18, 11, 11)
        obs = obs.unsqueeze(0)
        return self.agent_model(obs)
