import torch
import torch.nn as nn


class AgentLoss(nn.Module):

    def __init__(self, gamma, value_coef, entropy_coef, gae_lambda, value_loss_coef):
        super().__init__()
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.gae_lambda = gae_lambda
        self.value_loss_coef = value_loss_coef
        self.opponent_loss_func = nn.CrossEntropyLoss()

    def forward(self, R, gamma, agent_rewards, agent_log_probs, agent_values, agent_entropies, opponent_log_probs, opponent_ground_truths):
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(agent_rewards))):
            R = gamma * R + agent_rewards[i]
            advantage = R - agent_values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)
            # Generalized Advantage Estimation
            delta_t = agent_rewards[i] + gamma * agent_values[i + 1] - agent_values[i]
            gae = gae * gamma * self.gae_lambda + delta_t
            policy_loss = policy_loss - agent_log_probs[i] * gae.detach() - self.entropy_coef * agent_entropies[i]
        a3c_loss = policy_loss + self.value_loss_coef * value_loss
        opponent_log_probs = torch.stack(opponent_log_probs)
        opponent_ground_truths = torch.stack(opponent_ground_truths)
        opponent_loss = self.opponent_loss_func(opponent_log_probs, opponent_ground_truths)
        opponent_loss = opponent_loss.mean()
        total_loss = a3c_loss + opponent_loss
        return total_loss
