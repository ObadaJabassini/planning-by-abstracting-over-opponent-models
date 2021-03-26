import torch
import torch.nn as nn


class AgentModel(nn.Module):
    def __init__(self,
                 features_extractor,
                 agent_nb_actions,
                 nb_opponents,
                 opponent_nb_actions,
                 nb_units,
                 nb_attention_heads=4):
        super().__init__()
        self.features_extractor = features_extractor
        features_size = self.features_extractor.output_size
        self.agent_latent_layer = nn.Sequential(
            nn.Linear(features_size, nb_units),
            nn.ELU()
        )
        self.multihead_attention = nn.MultiheadAttention(embed_dim=nb_units, num_heads=nb_attention_heads)
        self.agent_head_layer = nn.Sequential(
            nn.Linear(nb_units * 2, nb_units),
            nn.ELU()
        )
        self.agent_policy_layer = nn.Linear(nb_units, agent_nb_actions)
        self.agent_value_layer = nn.Sequential(
            nn.Linear(nb_units, 1),
        )
        opponent_latent_layers = [nn.Sequential(nn.Linear(features_size, nb_units), nn.ELU()) for _ in
                                  range(nb_opponents)]
        self.opponent_latent_layers = nn.ModuleList(opponent_latent_layers)
        opponent_head_layers = [nn.Sequential(nn.Linear(nb_units, nb_units), nn.ELU()) for _ in range(nb_opponents)]
        self.opponent_head_layers = nn.ModuleList(opponent_head_layers)
        opponent_policies_layers = [nn.Linear(nb_units, opponent_nb_actions) for _ in range(nb_opponents)]
        self.opponent_policies_layers = nn.ModuleList(opponent_policies_layers)
        opponent_values_layers = [nn.Sequential(nn.Linear(nb_units, 1)) for _ in range(nb_opponents)]
        self.opponent_values_layers = nn.ModuleList(opponent_values_layers)

    def forward(self, image):
        features = self.features_extractor(image)
        agent_latent = self.agent_latent_layer(features)
        opponent_latents = [opponent_latent_layer(features) for opponent_latent_layer in self.opponent_latent_layers]

        # attention mechanism

        # (nb_opponents, batch_size, latent_dim)
        opponent_latents_stacked = torch.stack(opponent_latents)
        # (1, batch_size, latent_dim)
        agent_latent_stacked = agent_latent.unsqueeze(0)
        attn_output, attn_output_weights = self.multihead_attention(agent_latent_stacked,
                                                                    opponent_latents_stacked,
                                                                    opponent_latents_stacked)
        # (batch_size, nb_opponents), will be used later for planning
        opponents_influences = attn_output_weights.squeeze(1)
        # back to (batch_size, latent_dim)
        attn_output = attn_output.squeeze(0)
        # (batch_size, latent_dim * 2)
        agent_latent = torch.cat((agent_latent, attn_output), dim=1)

        # output
        agent_head = self.agent_head_layer(agent_latent)
        agent_policy = self.agent_policy_layer(agent_head)
        agent_value = self.agent_value_layer(agent_head)
        opponents_heads = [self.opponent_head_layers[i](opponent_latents[i]) for i in range(len(opponent_latents))]
        opponents_policies = [self.opponent_policies_layers[i](opponents_heads[i]) for i in range(len(opponents_heads))]
        opponents_policies = torch.stack(opponents_policies, 1)
        opponents_values = [self.opponent_values_layers[i](opponents_heads[i]) for i in range(len(opponents_heads))]
        opponents_values = torch.stack(opponents_values, 1)
        return agent_policy, agent_value, opponents_policies, opponents_values, opponents_influences
