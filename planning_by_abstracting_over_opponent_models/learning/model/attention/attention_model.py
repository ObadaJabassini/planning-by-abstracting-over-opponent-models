import torch
import torch.nn as nn

from planning_by_abstracting_over_opponent_models.learning.model.attention.hard_attention import HardAttention
from planning_by_abstracting_over_opponent_models.learning.model.attention.multihead_soft_attention import \
    MultiheadSoftAttention


class AttentionModel(nn.Module):

    def __init__(self,
                 latent_dim,
                 nb_soft_attention_heads,
                 hard_attention_rnn_hidden_size,
                 approximate_hard_attention=True):
        super().__init__()
        self.nb_soft_attention_heads = nb_soft_attention_heads
        if self.nb_soft_attention_heads is not None:
            self.hard_attention = HardAttention(latent_dim=latent_dim,
                                                hard_attention_rnn_hidden_size=hard_attention_rnn_hidden_size,
                                                approximate=approximate_hard_attention)
            self.multihead_soft_attention = MultiheadSoftAttention(latent_dim=latent_dim,
                                                                   embed_dim=latent_dim,
                                                                   nb_heads=nb_soft_attention_heads)

    def forward(self, agent_latent, opponent_latents):
        if self.nb_soft_attention_heads is None:
            for opponent_latent in opponent_latents:
                agent_latent = agent_latent * opponent_latent
            ll = len(opponent_latents) + 1
            scores = torch.full((agent_latent.shape[0], ll), 1 / ll).to(agent_latent.device)
            return agent_latent, scores
        # (nb_opponents, batch_size, latent_dim)
        opponent_latents = torch.stack(opponent_latents, dim=0)
        hard_attention = self.hard_attention(agent_latent, opponent_latents)
        attention_output, attention_scores = self.multihead_soft_attention(agent_latent=agent_latent,
                                                                           opponent_latents=opponent_latents,
                                                                           hard_attention=hard_attention)
        return attention_output, attention_scores
