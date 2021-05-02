from typing import List
import numpy as np
import pommerman
import torch

from planning_by_abstracting_over_opponent_models.agent import Agent
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


def check_agent_existence(state, board_tuple, index):
    result = np.zeros(board_tuple)
    idd = 10 + index
    if idd in state[0]["alive"]:
        position = state[index]["position"]
        result[position] = 1
    return result


def extract_features(state, nb_opponents, max_steps):
    agent_state = state[0]
    bomb_blast_strength_map = agent_state["bomb_blast_strength"]
    board_size = bomb_blast_strength_map.shape[0]
    board_tuple = (board_size, board_size)
    bomb_life_map = agent_state["bomb_life"]
    agent_position_map = check_agent_existence(state, board_tuple, 0)
    ammo_map = np.full(board_tuple, agent_state["ammo"])
    blast_strength_map = np.full(board_tuple, agent_state["blast_strength"])
    can_kick_map = np.full(board_tuple, int(agent_state["can_kick"]))
    teammate_existence_map = np.zeros(board_tuple)
    if nb_opponents == 1:
        opponents_position_map = [check_agent_existence(state, board_tuple, 1),
                                  np.zeros(board_tuple),
                                  np.zeros(board_tuple)]
    else:
        opponents_position_map = [check_agent_existence(state, board_tuple, 2),
                                  check_agent_existence(state, board_tuple, 1),
                                  check_agent_existence(state, board_tuple, 3)]

    board = agent_state["board"]
    passage_position_map = (board == 0).astype(int)
    rigid_wall_position_map = (board == 1).astype(int)
    wood_wall_position_map = (board == 2).astype(int)
    flames_position_map = (board == 3).astype(int)
    extra_bomb_position_map = (board == 6).astype(int)
    incr_range_position_map = (board == 7).astype(int)
    kick_position_map = (board == 8).astype(int)
    current_step_map = np.full(board_tuple, agent_state["step_count"]).astype(float) / max_steps
    maps = [
        bomb_blast_strength_map,
        bomb_life_map,
        agent_position_map,
        ammo_map,
        blast_strength_map,
        can_kick_map,
        teammate_existence_map,
        *opponents_position_map,
        passage_position_map,
        rigid_wall_position_map,
        wood_wall_position_map,
        flames_position_map,
        extra_bomb_position_map,
        incr_range_position_map,
        kick_position_map,
        current_step_map
    ]
    features = np.stack(maps, axis=0)
    return features


def get_observation(state, nb_opponents, max_steps, device):
    features = extract_features(state, nb_opponents, max_steps)
    # (18, 11, 11)
    obs = torch.from_numpy(features)
    # (18, 11, 11), swap width and height
    obs = obs.permute(0, 2, 1)
    obs = obs.float().unsqueeze(0).to(device)
    return obs