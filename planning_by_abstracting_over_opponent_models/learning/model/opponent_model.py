import torch.nn as nn


class OpponentModel(nn.Module):
    def __init__(self, features_size, latent_dim, head_dim, nb_actions):
        super().__init__()
        self.latent_layer = nn.Sequential(nn.Linear(features_size, latent_dim), nn.ELU())
        self.head_layer = nn.Sequential(nn.Linear(latent_dim, head_dim), nn.ELU())
        self.policy_layer = nn.Linear(head_dim, nb_actions)
        self.value_layer = nn.Linear(head_dim, 1)

    def forward(self, features):
        latent = self.latent_layer(features)
        head = self.head_layer(latent)
        policy = self.policy_layer(head)
        value = self.value_layer(head)
        return latent, policy, value
