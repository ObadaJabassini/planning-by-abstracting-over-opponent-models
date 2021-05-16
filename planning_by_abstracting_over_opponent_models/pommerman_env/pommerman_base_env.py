import abc
import numpy as np

import torch


class PommermanBaseEnv(abc.ABC):

    def __init__(self, nb_players):
        self.board_size = 11
        self.max_steps = 1000
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
        rewards = (rewards + 1) / 2
        return rewards

    def get_agent_position_map(self, state, index):
        result = np.zeros((self.board_size, self.board_size))
        idd = 10 + index
        if idd in state[index]["alive"]:
            position = state[index]["position"]
            result[position] = 1
        return result

    def get_features(self, state):
        board_tuple = (self.board_size, self.board_size)
        features = np.zeros((self.nb_players, 18, self.board_size, self.board_size))
        agent_position_maps = [self.get_agent_position_map(state, i) for i in range(self.nb_players)]
        board = state[0]["board"]
        passage_position_map = (board == 0).astype(int)
        rigid_wall_position_map = (board == 1).astype(int)
        wood_wall_position_map = (board == 2).astype(int)
        flames_position_map = (board == 3).astype(int)
        extra_bomb_position_map = (board == 6).astype(int)
        incr_range_position_map = (board == 7).astype(int)
        kick_position_map = (board == 8).astype(int)
        current_step_map = np.full(board_tuple, state[0]["step_count"] / self.max_steps)
        for agent_id in range(self.nb_players):
            agent_state = state[agent_id]
            bomb_blast_strength_map = agent_state["bomb_blast_strength"]
            bomb_life_map = agent_state["bomb_life"]
            agent_position_map = agent_position_maps[agent_id]
            ammo_map = np.full(board_tuple, agent_state["ammo"])
            blast_strength_map = np.full(board_tuple, agent_state["blast_strength"])
            can_kick_map = np.full(board_tuple, int(agent_state["can_kick"]))
            teammate_existence_map = np.zeros(board_tuple)
            if self.nb_players == 2:
                idx = 1 if agent_id == 0 else 1
                opponents_position_map = [agent_position_maps[idx],
                                          np.zeros(board_tuple),
                                          np.zeros(board_tuple)]
            else:
                if agent_id == 0:
                    i, j, k = 2, 1, 3
                elif agent_id == 1:
                    i, j, k = 3, 2, 0
                elif agent_id == 2:
                    i, j, k = 0, 3, 1
                else:
                    i, j, k = 1, 0, 2
                opponents_position_map = [agent_position_maps[i],
                                          agent_position_maps[j],
                                          agent_position_maps[k]]
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
            features_map = np.stack(features_map, axis=0)
            features[agent_id] = features_map
        features = torch.from_numpy(features).float()
        # (nb_players, 18, 11, 11), swap width and height
        features = features.permute(0, 1, 3, 2)
        return features

