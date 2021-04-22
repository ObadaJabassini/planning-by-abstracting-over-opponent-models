import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic

from planning_by_abstracting_over_opponent_models.learning.soft_hard_multihead_attention import SoftHardMultiheadAttention


class AgentModel(nn.Module):
    def __init__(self,
                 features_extractor,
                 agent_nb_actions,
                 nb_opponents,
                 opponent_nb_actions,
                 head_dim,
                 latent_dim,
                 nb_soft_attention_heads=None,
                 hard_attention_rnn_hidden_size=None):
        super().__init__()
        self.features_extractor = features_extractor
        self.nb_opponents = nb_opponents
        self.nb_soft_attention_heads = nb_soft_attention_heads
        self.use_attention = nb_soft_attention_heads is not None
        self.use_hard_attention = hard_attention_rnn_hidden_size is not None
        features_size = self.features_extractor.output_size
        self.agent_latent_layer = nn.Sequential(
            nn.Linear(features_size, latent_dim),
            nn.ELU()
        )
        head_size = latent_dim
        if self.use_attention:
            if self.use_hard_attention:
                self.lstm = nn.LSTM(latent_dim * 2, hard_attention_rnn_hidden_size, bidirectional=True)
                self.hard_attention_layer = nn.Linear(hard_attention_rnn_hidden_size * 2, 2)
            self.multihead_attention = SoftHardMultiheadAttention(embed_dim=latent_dim, num_heads=nb_soft_attention_heads)
            head_size *= 2

        self.agent_head_layer = nn.Sequential(
            nn.Linear(head_size, head_dim),
            nn.ELU()
        )
        self.agent_policy_layer = nn.Linear(head_dim, agent_nb_actions)
        self.agent_value_layer = nn.Sequential(
            nn.Linear(head_dim, 1),
            nn.ELU()
        )
        opponent_latent_layers = [nn.Sequential(nn.Linear(features_size, latent_dim), nn.ELU()) for _ in
                                  range(nb_opponents)]
        self.opponent_latent_layers = nn.ModuleList(opponent_latent_layers)
        opponent_head_layers = [nn.Sequential(nn.Linear(latent_dim, head_dim), nn.ELU()) for _ in range(nb_opponents)]
        self.opponent_head_layers = nn.ModuleList(opponent_head_layers)
        opponent_policies_layers = [nn.Linear(head_dim, opponent_nb_actions) for _ in range(nb_opponents)]
        self.opponent_policies_layers = nn.ModuleList(opponent_policies_layers)
        opponent_values_layers = [nn.Sequential(nn.Linear(head_dim, 1), nn.ELU()) for _ in range(nb_opponents)]
        self.opponent_values_layers = nn.ModuleList(opponent_values_layers)

    def forward(self, obs):
        features = self.features_extractor(obs)
        agent_latent = self.agent_latent_layer(features)
        opponent_latents = [opponent_latent_layer(features) for opponent_latent_layer in self.opponent_latent_layers]

        if self.use_attention:
            # use the attention mechanism
            agent_latent, opponent_influence = self.attend(agent_latent, opponent_latents)
        else:
            for opponent_latent in opponent_latents:
                agent_latent = agent_latent * opponent_latent
            opponent_influence = torch.ones(agent_latent.size(0), self.nb_opponents, device=agent_latent.device) / self.nb_opponents

        # output
        agent_head = self.agent_head_layer(agent_latent)
        agent_policy = self.agent_policy_layer(agent_head)
        agent_value = self.agent_value_layer(agent_head)
        opponents_heads = [self.opponent_head_layers[i](opponent_latents[i]) for i in range(self.nb_opponents)]
        opponents_policies = [self.opponent_policies_layers[i](opponents_heads[i]) for i in range(self.nb_opponents)]
        opponents_policies = torch.stack(opponents_policies, 1)
        opponent_values = [self.opponent_values_layers[i](opponents_heads[i]) for i in range(self.nb_opponents)]
        opponent_values = torch.stack(opponent_values, 1)
        return agent_policy, agent_value, opponents_policies, opponent_values, opponent_influence

    def attend(self, agent_latent, opponent_latents):
        # (1, batch_size, latent_dim)
        agent_latent_stacked = agent_latent.unsqueeze(0)
        # (nb_opponents, batch_size, latent_dim)
        opponent_latents_stacked = torch.stack(opponent_latents)

        if self.use_hard_attention:
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
            hard_attention = F.gumbel_softmax(hard_attention, tau=0.01, hard=True, dim=-1)
            # (nb_opponents, batch_size)
            hard_attention = hard_attention[..., 1]
            # (batch_size, nb_opponents)
            hard_attention = hard_attention.T
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
