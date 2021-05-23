from typing import List

import pommerman
import torch

from planning_by_abstracting_over_opponent_models.learning.agent import Agent
from planning_by_abstracting_over_opponent_models.learning.agent_model import AgentModel
from planning_by_abstracting_over_opponent_models.learning.features_extractor import FeaturesExtractor
from planning_by_abstracting_over_opponent_models.pommerman_env.pommerman_base_env import PommermanBaseEnv
from planning_by_abstracting_over_opponent_models.pommerman_env.pommerman_cython_env import PommermanCythonEnv
from planning_by_abstracting_over_opponent_models.pommerman_env.pommerman_python_env import PommermanPythonEnv


def create_env(rank,
               seed,
               use_cython,
               model_spec,
               nb_actions,
               nb_opponents,
               opponent_class,
               device,
               train=True):
    agent_model = create_agent_model(rank,
                                     seed,
                                     nb_actions,
                                     nb_opponents,
                                     device=device,
                                     train=train,
                                     **model_spec)
    agent = Agent(0, agent_model)
    agents: List[pommerman.agents.BaseAgent] = [opponent_class() for _ in range(nb_opponents)]
    agents.insert(0, agent)
    r = seed + rank
    env: PommermanBaseEnv = PommermanCythonEnv(agents, r) if use_cython else PommermanPythonEnv(agents, r)
    return agents, env


def create_agent_model(rank,
                       seed,
                       nb_actions,
                       nb_opponents,
                       nb_conv_layers,
                       nb_filters,
                       latent_dim,
                       head_dim,
                       nb_soft_attention_heads,
                       hard_attention_rnn_hidden_size,
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
                             head_dim=head_dim,
                             latent_dim=latent_dim,
                             nb_soft_attention_heads=nb_soft_attention_heads,
                             hard_attention_rnn_hidden_size=hard_attention_rnn_hidden_size).to(device)
    agent_model.train(train)
    return agent_model

