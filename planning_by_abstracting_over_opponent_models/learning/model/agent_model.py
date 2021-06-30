import torch
import torch.nn as nn

from planning_by_abstracting_over_opponent_models.learning.model.attention.attention_model import AttentionModel
from planning_by_abstracting_over_opponent_models.learning.model.features_extractor import FeaturesExtractor
from planning_by_abstracting_over_opponent_models.learning.model.opponent_model import OpponentModel


class AgentModel(nn.Module):
    def __init__(self,
                 features_extractor,
                 agent_nb_actions,
                 nb_opponents,
                 opponent_nb_actions,
                 latent_dim,
                 nb_soft_attention_heads=4,
                 hard_attention_rnn_hidden_size=64,
                 approximate_hard_attention=True):
        super().__init__()
        self.features_extractor = features_extractor
        self.nb_soft_attention_heads = nb_soft_attention_heads
        features_size = self.features_extractor.output_size

        self.agent_latent_layer = nn.Sequential(
            nn.Linear(features_size, latent_dim),
            nn.ELU()
        )

        self.attention_model = AttentionModel(
            latent_dim=latent_dim,
            nb_soft_attention_heads=nb_soft_attention_heads,
            hard_attention_rnn_hidden_size=hard_attention_rnn_hidden_size,
            approximate_hard_attention=approximate_hard_attention)

        self.agent_head_layer = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ELU()
        )
        self.agent_policy_layer = nn.Linear(latent_dim, agent_nb_actions)
        self.agent_value_layer = nn.Linear(latent_dim, 1)

        opponent_models = [
            OpponentModel(features_size=features_size, latent_dim=latent_dim, nb_actions=opponent_nb_actions) for _ in
            range(nb_opponents)]
        self.opponent_models = nn.ModuleList(opponent_models)

    def forward(self, obs):
        features = self.features_extractor(obs)
        agent_latent = self.agent_latent_layer(features)
        opponent_outputs = [opponent_model(features) for opponent_model in self.opponent_models]
        opponent_latents, opponent_policies = list(zip(*opponent_outputs))
        agent_latent, opponent_influence = self.attention_model(agent_latent, opponent_latents)


        # output
        agent_head = self.agent_head_layer(agent_latent)
        agent_policy = self.agent_policy_layer(agent_head)
        agent_value = self.agent_value_layer(agent_head)

        opponent_policies = torch.stack(opponent_policies, dim=1)

        return agent_policy, agent_value, opponent_policies, opponent_influence


def create_agent_model(rank,
                       seed,
                       nb_actions,
                       nb_opponents,
                       nb_conv_layers,
                       nb_filters,
                       latent_dim,
                       nb_soft_attention_heads,
                       hard_attention_rnn_hidden_size,
                       approximate_hard_attention,
                       device,
                       train=True):
    torch.manual_seed(seed + rank)
    nb_filters = [nb_filters] * nb_conv_layers
    features_extractor = FeaturesExtractor(input_size=(11, 11, 18),
                                           nb_filters=nb_filters,
                                           filter_size=3,
                                           filter_stride=1,
                                           filter_padding=1)
    agent_model = AgentModel(features_extractor=features_extractor,
                             nb_opponents=nb_opponents,
                             agent_nb_actions=nb_actions,
                             opponent_nb_actions=nb_actions,
                             latent_dim=latent_dim,
                             nb_soft_attention_heads=nb_soft_attention_heads,
                             hard_attention_rnn_hidden_size=hard_attention_rnn_hidden_size,
                             approximate_hard_attention=approximate_hard_attention).to(device)
    agent_model.train(train)
    return agent_model
