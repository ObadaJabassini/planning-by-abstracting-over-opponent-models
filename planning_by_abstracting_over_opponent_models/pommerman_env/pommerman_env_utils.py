from typing import List

import pommerman
import torch

from planning_by_abstracting_over_opponent_models.pommerman_env.agent import Agent
from planning_by_abstracting_over_opponent_models.learning.agent_model import AgentModel
from planning_by_abstracting_over_opponent_models.learning.features_extractor import FeaturesExtractor


def create_env(seed,
               rank,
               device,
               model_spec,
               nb_actions,
               nb_opponents,
               max_steps,
               train=True):
    agent_model = create_agent_model(seed,
                                     rank,
                                     nb_actions,
                                     nb_opponents,
                                     device,
                                     train=train,
                                     **model_spec)
    agent = Agent(agent_model, nb_opponents, max_steps, device)
    agents: List[pommerman.agents.BaseAgent] = [pommerman.agents.SimpleAgent() for _ in range(nb_opponents)]
    agents.insert(0, agent)
    env = pommerman.make('PommeFFACompetition-v0', agents)
    env.seed(seed + rank)
    env.set_training_agent(0)
    return agents, env


def create_agent_model(seed,
                       rank,
                       nb_actions,
                       nb_opponents,
                       device,
                       nb_conv_layers,
                       nb_filters,
                       latent_dim,
                       head_dim,
                       nb_soft_attention_heads,
                       hard_attention_rnn_hidden_size,
                       train=True):
    torch.manual_seed(seed + rank)
    nb_filters = [nb_filters] * nb_conv_layers
    features_extractor = FeaturesExtractor(input_size=(11, 11, 18),
                                           nb_filters=nb_filters,
                                           filter_size=3,
                                           filter_stride=1,
                                           filter_padding=1).to(device)

    agent_model = AgentModel(features_extractor=features_extractor,
                             nb_opponents=nb_opponents,
                             agent_nb_actions=nb_actions,
                             opponent_nb_actions=nb_actions,
                             head_dim=head_dim,
                             latent_dim=latent_dim,
                             nb_soft_attention_heads=nb_soft_attention_heads,
                             hard_attention_rnn_hidden_size=hard_attention_rnn_hidden_size)
    agent_model = agent_model.to(device)
    agent_model.train(train)
    return agent_model

