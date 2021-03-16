import torch
import torch.nn as nn


class AgentModel(nn.Module):
    def __init__(self, features_extractor, agent_nb_actions, nb_opponents, opponent_nb_actions, nb_units):
        super().__init__()
        self.features_extractor = features_extractor
        features_size = self.features_extractor.output_size
        self.agent_latent_layer = nn.Sequential(
            nn.Linear(features_size, nb_units),
            nn.ELU()
        )
        self.agent_head_layer = nn.Sequential(
            nn.Linear(nb_units, nb_units),
            nn.ELU()
        )
        self.agent_policy_layer = nn.Linear(nb_units, agent_nb_actions)
        self.agent_value_layer = nn.Sequential(
            nn.Linear(nb_units, 1),
            # should add a activation function (e.g. absolute)
        )
        self.opponent_latent_layers = nn.ModuleList([nn.Sequential(nn.Linear(features_size, nb_units), nn.ELU())] * nb_opponents)
        self.opponent_head_layers = nn.ModuleList([nn.Sequential(nn.Linear(nb_units, nb_units), nn.ELU())] * nb_opponents)
        self.opponent_policies_layers = nn.ModuleList([nn.Linear(nb_units, opponent_nb_actions)] * nb_opponents)
        self.opponent_values_layers = nn.ModuleList([nn.Sequential(nn.Linear(nb_units, 1))] * nb_opponents)

    def forward(self, image):
        features = self.features_extractor(image)
        agent_latent = self.agent_latent_layer(features)
        opponent_latents = [opponent_latent_layer(features) for opponent_latent_layer in self.opponent_latent_layers]
        for opponent_latent in opponent_latents:
            agent_latent *= opponent_latent
        agent_head = self.agent_head_layer(agent_latent)
        agent_policy = self.agent_policy_layer(agent_head)
        agent_value = self.agent_value_layer(agent_head)
        opponents_heads = [self.opponent_head_layers[i](opponent_latents[i]) for i in range(len(opponent_latents))]
        opponents_policies = [self.opponent_policies_layers[i](opponents_heads[i]) for i in range(len(opponents_heads))]
        opponents_policies = torch.stack(opponents_policies, 1)
        opponents_values = [self.opponent_values_layers[i](opponents_heads[i]) for i in range(len(opponents_heads))]
        opponents_values = torch.stack(opponents_values, 1)
        return agent_policy, agent_value, opponents_policies, opponents_values
