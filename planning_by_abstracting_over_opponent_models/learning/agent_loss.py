# partially inspired from https://github.com/ikostrikov/pytorch-a3c/blob/master/train.py


import torch
import torch.nn as nn
import torch.nn.functional as F


class AgentLoss(nn.Module):

    def __init__(self, gamma, value_coef, entropy_coef, gae_lambda, value_loss_coef):
        super().__init__()
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.gae_lambda = gae_lambda
        self.value_loss_coef = value_loss_coef

    def agent_loss_func(self, R, agent_rewards, agent_values, agent_log_probs, agent_entropies):
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1).to(agent_entropies[0].device)
        for i in reversed(range(len(agent_rewards))):
            R = self.gamma * R + agent_rewards[i]
            advantage = R - agent_values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)
            # Generalized Advantage Estimation
            delta_t = agent_rewards[i] + self.gamma * agent_values[i + 1] - agent_values[i]
            gae = gae * self.gamma * self.gae_lambda + delta_t
            policy_loss = policy_loss - agent_log_probs[i] * gae.detach() - self.entropy_coef * agent_entropies[i]
        total_loss = policy_loss + self.value_loss_coef * value_loss
        return total_loss

    def opponent_loss_func(self,
                           opponent_log_probs,
                           opponent_actions_ground_truths,
                           opponent_values,
                           opponent_rewards,
                           opponent_coefs):
        opponent_coef = torch.FloatTensor(opponent_coefs).to(opponent_log_probs.device)
        policy_loss = F.cross_entropy(opponent_log_probs, opponent_actions_ground_truths, reduction='none')
        policy_loss = opponent_coef * policy_loss
        policy_loss = policy_loss.mean()
        shifted_dim = opponent_values.shape[0] - 1
        shifted_values = torch.roll(opponent_values, -1, 0)
        shifted_values = shifted_values[:shifted_dim]
        predicted_values = opponent_rewards + self.gamma * shifted_values
        opponent_values = opponent_values[:shifted_dim]
        value_loss = F.smooth_l1_loss(opponent_values, predicted_values)
        total_loss = policy_loss + self.value_loss_coef * value_loss
        return total_loss

    def forward(self,
                R,
                agent_rewards,
                agent_log_probs,
                agent_values,
                agent_entropies,
                opponent_log_probs,
                opponent_actions_ground_truths,
                opponent_values,
                opponent_rewards,
                opponent_coefs):
        agent_loss = self.agent_loss_func(R, agent_rewards, agent_values, agent_log_probs, agent_entropies)
        opponent_loss = self.opponent_loss_func(opponent_log_probs,
                                                opponent_actions_ground_truths,
                                                opponent_values,
                                                opponent_rewards,
                                                opponent_coefs)
        total_loss = agent_loss + opponent_loss
        return total_loss
