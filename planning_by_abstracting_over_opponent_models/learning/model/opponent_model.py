import torch.nn as nn


class OpponentModel(nn.Module):
    def __init__(self, features_size, latent_dim, nb_actions):
        super().__init__()
        self.latent_layer = nn.Sequential(nn.Linear(features_size, latent_dim), nn.ELU())
        self.policy_layer = nn.Linear(latent_dim, nb_actions)

    def forward(self, features):
        latent = self.latent_layer(features)
        policy = self.policy_layer(latent)
        return latent, policy
