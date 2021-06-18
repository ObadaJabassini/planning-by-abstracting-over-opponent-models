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
        self.hard_attention = HardAttention(latent_dim=latent_dim,
                                            hard_attention_rnn_hidden_size=hard_attention_rnn_hidden_size,
                                            approximate=approximate_hard_attention) if hard_attention_rnn_hidden_size is not None else None
        self.multihead_soft_attention = MultiheadSoftAttention(latent_dim=latent_dim,
                                                               embed_dim=latent_dim,
                                                               nb_heads=nb_soft_attention_heads)
        self.embedding_agent_layer = nn.Linear(latent_dim, latent_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.embedding_agent_layer.weight)

    def forward(self, agent_latent, opponent_latents):
        if self.hard_attention is not None:
            hard_attention = self.hard_attention(agent_latent, opponent_latents)
        else:
            hard_attention = torch.ones(agent_latent.view(0), len(opponent_latents), device=agent_latent.device)

        # soft attention
        attention_output, attention_scores = self.multihead_soft_attention.forward(agent_latent=agent_latent,
                                                                                   opponent_latents=opponent_latents,
                                                                                   hard_attention=hard_attention)
        agent_latent = self.embedding_agent_layer(agent_latent)
        agent_latent = agent_latent + attention_output
        return agent_latent, attention_scores
