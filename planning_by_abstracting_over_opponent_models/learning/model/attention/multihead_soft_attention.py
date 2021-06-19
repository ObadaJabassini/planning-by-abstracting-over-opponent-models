from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftAttention(nn.Module):
    def __init__(self, latent_dim, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.w_s = nn.Linear(latent_dim, embed_dim, bias=False)
        self.w_t = nn.Linear(latent_dim, embed_dim, bias=False)
        self.w_c = nn.Linear(latent_dim, embed_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.w_s.weight)
        nn.init.xavier_normal_(self.w_t.weight)
        nn.init.xavier_normal_(self.w_c.weight)

    # agent_latent: (batch_size, latent_dim)
    # opponent_latent (batch_size, nb_opponents, latent_dim)
    # hard attention (batch_size, nb_opponents)
    def forward(self, agent_latent, opponent_latents, hard_attention):
        # (batch_size, embed_dim, 1)
        agent_latent = self.w_s(agent_latent).unsqueeze(2)
        # (batch_size, nb_opponents, embed_dim)
        embedded_opponent_latents = self.w_t(opponent_latents)
        # (batch_size, nb_opponents)
        scores = torch.bmm(embedded_opponent_latents, agent_latent).squeeze(2)
        scores = F.softmax(scores, dim=-1)
        scores = scores * hard_attention
        # (batch_size, nb_opponents, embed_dim)
        opponent_latents = self.w_c(opponent_latents)
        # (batch_size, nb_opponents, embed_dim)
        repeated_scores = scores.unsqueeze(2).repeat(1, 1, self.embed_dim)
        opponent_latents = repeated_scores * opponent_latents
        # (batch_size, embed_dim)
        result = torch.sum(opponent_latents, dim=1)
        return result, scores


class MultiheadSoftAttention(nn.Module):
    def __init__(self, latent_dim, embed_dim, nb_heads):
        super().__init__()
        self.heads = nn.ModuleList([SoftAttention(latent_dim, embed_dim) for _ in range(nb_heads)])

    def forward(self, agent_latent, opponent_latents, hard_attention) -> Tuple[torch.Tensor, torch.Tensor]:
        # (batch_size, nb_opponents, latent_dim)
        opponent_latents = torch.stack(opponent_latents, dim=1)
        outputs = [head(agent_latent, opponent_latents, hard_attention) for head in self.heads]
        embeddings, scores = list(zip(*outputs))
        embeddings = torch.mean(torch.stack(embeddings), dim=0)
        scores = torch.mean(torch.stack(scores), dim=0)
        return embeddings, scores
