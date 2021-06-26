import abc
import numpy as np

import torch
from pommerman.constants import Item


class PommermanBaseEnv(abc.ABC):

    def __init__(self, nb_players):
        self.board_size = 11
        self.max_steps = 800
        self.nb_players = nb_players

    @abc.abstractmethod
    def get_observations(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_done(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_rewards(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def step(self, actions):
        raise NotImplementedError()

    @abc.abstractmethod
    def act(self, state):
        raise NotImplementedError()

    @abc.abstractmethod
    def render(self, mode=None):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_game_state(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def set_game_state(self, game_state):
        raise NotImplementedError()

    def transform_rewards(self, rewards):
        rewards = np.asarray(rewards[:self.nb_players])
        return rewards

    def get_agent_position_map(self, state, index):
        result = np.zeros((self.board_size, self.board_size)).astype(float)
        idd = 10 + index
        if idd in state[0]["alive"]:
            position = state[index]["position"]
            result[position] = 1
        return result

    def get_features(self, state):
        board_tuple = (self.board_size, self.board_size)
        features = np.zeros((self.nb_players, 18, self.board_size, self.board_size))
        agent_position_maps = [self.get_agent_position_map(state, i) for i in range(self.nb_players)]
        board = state[0]["board"]
        passage_position_map = (board == Item.Passage.value).astype(float)
        rigid_wall_position_map = (board == Item.Rigid.value).astype(float)
        wood_wall_position_map = (board == Item.Wood.value).astype(float)
        flames_position_map = (board == Item.Flames.value).astype(float)
        extra_bomb_position_map = (board == Item.ExtraBomb.value).astype(float)
        incr_range_position_map = (board == Item.IncrRange.value).astype(float)
        kick_position_map = (board == Item.Kick.value).astype(float)
        current_step_map = np.full(board_tuple, state[0]["step_count"] / self.max_steps).astype(float)
        for agent_id in range(self.nb_players):
            agent_state = state[agent_id]
            bomb_blast_strength_map = agent_state["bomb_blast_strength"].astype(float)
            bomb_life_map = agent_state["bomb_life"].astype(float)
            agent_position_map = agent_position_maps[agent_id]
            ammo_map = np.full(board_tuple, agent_state["ammo"]).astype(float)
            blast_strength_map = np.full(board_tuple, agent_state["blast_strength"]).astype(float)
            can_kick_map = np.full(board_tuple, int(agent_state["can_kick"])).astype(float)
            teammate_existence_map = np.zeros(board_tuple).astype(float)
            next_index, opposite_index, prev_index = (agent_id + 1) % 4, (agent_id + 2) % 4, (agent_id + 3) % 4
            opponents_position_map = [agent_position_maps[next_index],
                                      agent_position_maps[opposite_index],
                                      agent_position_maps[prev_index]]
            features_map = [
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
            features[agent_id] = np.stack(features_map, axis=0).astype(float)
        features = torch.from_numpy(features).float()
        # (nb_players, 18, 11, 11), swap width and height
        features = features.permute(0, 1, 3, 2)
        return features

