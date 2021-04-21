# partially inspired by https://github.com/ikostrikov/pytorch-a3c/blob/master/train.py

from icecream import ic
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

    def agent_loss_func(self, agent_rewards, agent_values, agent_log_probs, agent_entropies):
        policy_loss = 0
        value_loss = 0
        idx = len(agent_values) - 1
        R = agent_values[idx]
        gae = torch.zeros(1).to(agent_entropies[0].device)
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
        """

        :param opponent_log_probs: (nb_opponents, nb_steps, nb_actions)
        :param opponent_actions_ground_truths: (nb_opponents, nb_steps)
        :param opponent_values: (nb_opponents, nb_steps)
        :param opponent_rewards: (nb_opponents, nb_steps)
        :param opponent_coefs: (nb_opponents)
        :return:
        """
        nb_opponents = opponent_log_probs.shape[0]
        total_loss = 0
        for i in range(nb_opponents):
            # policy loss
            policy_loss = opponent_coefs[i] * F.cross_entropy(opponent_log_probs[i], opponent_actions_ground_truths[i])

            # value loss
            opponent_value = opponent_values[i]
            opponent_reward = opponent_rewards[i]
            shifted_value = torch.roll(opponent_value, -1)
            shifted_dim = opponent_value.shape[0] - 1
            opponent_value = opponent_value[:shifted_dim]
            shifted_value = shifted_value[:shifted_dim]
            predicted_values = opponent_reward + self.gamma * shifted_value
            value_loss = opponent_coefs[i] * F.smooth_l1_loss(opponent_value, predicted_values)

            total_loss = total_loss + policy_loss + value_loss

        return total_loss

    def forward(self,
                agent_rewards,
                agent_log_probs,
                agent_values,
                agent_entropies,
                opponent_log_probs,
                opponent_actions_ground_truths,
                opponent_values,
                opponent_rewards,
                opponent_coefs):
        agent_loss = self.agent_loss_func(agent_rewards,
                                          agent_values,
                                          agent_log_probs,
                                          agent_entropies)

        opponent_loss = self.opponent_loss_func(opponent_log_probs,
                                                opponent_actions_ground_truths,
                                                opponent_values,
                                                opponent_rewards,
                                                opponent_coefs)

        total_loss = agent_loss + opponent_loss
        total_loss = total_loss.sum()
        return total_loss
