import torch
import torch.nn as nn
import torch.nn.functional as F

from planning_by_abstracting_over_opponent_models.learning.model.soft_hard_multihead_attention import \
    SoftHardMultiheadAttention


class AttentionModel(nn.Module):

    def __init__(self, nb_opponents, latent_dim, nb_soft_attention_heads, hard_attention_rnn_hidden_size, approximate_hard_attention=True):
        super().__init__()
        self.nb_opponents = nb_opponents
        self.use_hard_attention = hard_attention_rnn_hidden_size is not None
        if self.use_hard_attention:
            self.approximate_hard_attention = approximate_hard_attention
            self.lstm = nn.LSTM(latent_dim * 2, hard_attention_rnn_hidden_size, bidirectional=True)
            self.hard_attention_layer = nn.Linear(hard_attention_rnn_hidden_size * 2, 2)
        self.multihead_attention = SoftHardMultiheadAttention(embed_dim=latent_dim, num_heads=nb_soft_attention_heads)

    def forward(self, agent_latent, opponent_latents):
        # (1, batch_size, latent_dim)
        agent_latent_stacked = agent_latent.unsqueeze(0)
        # (nb_opponents, batch_size, latent_dim)
        opponent_latents_stacked = torch.stack(opponent_latents)

        if self.use_hard_attention:
            hard_attention = self.compute_hard_attention(agent_latent_stacked, opponent_latents_stacked)
        else:
            hard_attention = torch.ones(agent_latent.view(0), self.nb_opponents, device=agent_latent.device)

        # soft attention
        attn_output, attn_output_weights = self.multihead_attention(query=agent_latent_stacked,
                                                                    key=opponent_latents_stacked,
                                                                    hard_attention=hard_attention)
        # back to (batch_size, latent_dim)
        attn_output = attn_output.squeeze(0)
        # (batch_size, latent_dim * 2)
        agent_latent = torch.cat((agent_latent, attn_output), dim=-1)
        # (batch_size, nb_opponents), will be used later for planning
        opponent_influence = attn_output_weights.squeeze(1)
        return agent_latent, opponent_influence

    def compute_hard_attention(self, agent_latent_stacked, opponent_latents_stacked):
        # hard attention
        # (nb_opponents, batch_size, latent_dim)
        agent_latent_stacked_repeated = agent_latent_stacked.repeat(self.nb_opponents, 1, 1)
        # (nb_opponents, batch_size, latent_dim * 2)
        agent_opponent_stacked = torch.cat((agent_latent_stacked_repeated, opponent_latents_stacked), dim=-1)
        # (nb_opponents, batch_size, hard_attention_rnn_hidden_size * 2)
        lstm_output, _ = self.lstm(agent_opponent_stacked)
        # (nb_opponents, batch_size, 2)
        hard_attention = self.hard_attention_layer(lstm_output)
        # (nb_opponents, batch_size, 2)
        t = not self.approximate_hard_attention
        hard_attention = F.gumbel_softmax(hard_attention, tau=0.01, hard=t, dim=-1)
        # (nb_opponents, batch_size)
        hard_attention = hard_attention[..., 1]
        # (batch_size, nb_opponents)
        hard_attention = hard_attention.T
        return hard_attention
