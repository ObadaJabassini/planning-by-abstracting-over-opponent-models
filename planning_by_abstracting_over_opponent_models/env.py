from typing import List

import pommerman
import torch
from pommerman.agents import BaseAgent

from planning_by_abstracting_over_opponent_models.agent import Agent
from planning_by_abstracting_over_opponent_models.learning.agent_model import AgentModel
from planning_by_abstracting_over_opponent_models.learning.features_extractor import FeaturesExtractor


def create_env(seed, rank, device, action_space_size, nb_opponents, max_steps, train=True):
    agent, agent_model = create_agent_model(seed, rank, action_space_size, nb_opponents, max_steps, device, train=train)
    agents: List[BaseAgent] = [pommerman.agents.SimpleAgent() for _ in range(nb_opponents)]
    agents.insert(0, agent)
    env = pommerman.make('PommeFFACompetition-v0', agents)
    env.seed(seed + rank)
    env.set_training_agent(0)
    return agents, agent_model, env


def create_agent_model(seed,
                       rank,
                       action_space_size,
                       nb_opponents,
                       max_steps,
                       device,
                       nb_filters=None,
                       latent_dim=64,
                       head_dim=64,
                       nb_soft_attention_heads=None,
                       hard_attention_rnn_hidden_size=None,
                       train=True,
                       return_agent=True):
    torch.manual_seed(seed + rank)
    if nb_filters is None:
        nb_filters = [32, 32, 32]
    board_size = 11
    features_extractor = FeaturesExtractor(input_size=(board_size, board_size, 18),
                                           nb_filters=nb_filters,
                                           filter_size=3,
                                           filter_stride=1,
                                           filter_padding=1)

    agent_model = AgentModel(features_extractor=features_extractor,
                             nb_opponents=nb_opponents,
                             agent_nb_actions=action_space_size,
                             opponent_nb_actions=action_space_size,
                             head_dim=head_dim,
                             latent_dim=latent_dim,
                             nb_soft_attention_heads=nb_soft_attention_heads,
                             hard_attention_rnn_hidden_size=hard_attention_rnn_hidden_size)
    agent_model = agent_model.to(device)
    agent_model.train(train)
    if return_agent:
        agent = Agent(agent_model, nb_opponents, max_steps)
        return agent, agent_model
    return agent_model
