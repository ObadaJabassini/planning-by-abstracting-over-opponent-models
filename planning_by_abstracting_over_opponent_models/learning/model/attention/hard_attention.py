import torch
import torch.nn as nn
import torch.nn.functional as F


class HardAttention(nn.Module):
    def __init__(self, latent_dim, hard_attention_rnn_hidden_size, approximate=True):
        super().__init__()
        self.hard_attention_rnn_hidden_size = hard_attention_rnn_hidden_size
        if self.hard_attention_rnn_hidden_size is not None:
            self.hard = not approximate
            self.lstm = nn.LSTM(latent_dim * 2, hard_attention_rnn_hidden_size, bidirectional=True)
            self.output_layer = nn.Linear(hard_attention_rnn_hidden_size * 2, 2)

    def forward(self, agent_latent, opponent_latents):
        """
        :param agent_latent: (batch_size, latent_dim)
        :param opponent_latents: (nb_opponents, batch_size, latent_dim)
        :return:
        """
        if self.hard_attention_rnn_hidden_size is not None:
            nb_opponents = opponent_latents.shape[0]
            # (nb_opponents, batch_size, latent_dim)
            agent_latent_stacked_repeated = agent_latent.unsqueeze(0).repeat(nb_opponents, 1, 1)
            # (nb_opponents, batch_size, latent_dim * 2)
            agent_opponent_stacked = torch.cat((agent_latent_stacked_repeated, opponent_latents), dim=-1)
            # (nb_opponents, batch_size, hard_attention_rnn_hidden_size * 2)
            lstm_output, _ = self.lstm(agent_opponent_stacked)
            # (batch_size, nb_opponents, hard_attention_rnn_hidden_size * 2)
            lstm_output = lstm_output.permute(1, 0, 2)
            # (batch_size, nb_opponents, 2)
            hard_attention = self.output_layer(lstm_output)
            # (batch_size, nb_opponents, 2)
            hard_attention = F.gumbel_softmax(hard_attention, tau=0.01, hard=self.hard, dim=-1)
            # (batch_size, nb_opponents)
            hard_attention = hard_attention[..., 1]
            return hard_attention
        return torch.ones(agent_latent.shape[0], opponent_latents.shape[0], device=agent_latent.device)

