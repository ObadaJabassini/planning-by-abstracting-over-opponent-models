import numpy as np
import pommerman
import cpommerman
import torch

from planning_by_abstracting_over_opponent_models.config import cpu


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
    return features_map


def extract_observation(state, nb_opponents, max_steps):
    features = extract_features(state, nb_opponents, max_steps)
    # (18, 11, 11)
    obs = torch.from_numpy(features)
    # (18, 11, 11), swap width and height
    obs = obs.permute(0, 2, 1)
    obs = obs.float().unsqueeze(0)
    return obs


class PommermanWrappedEnv:
    def __init__(self, use_cython, agents, seed):
        self.use_cython = use_cython
        self.agents = agents
        self.opponents = self.agents[1:]
        if use_cython:
            self.env = cpommerman.make()
        else:
            self.env = pommerman.make('PommeFFACompetition-v0', agents)
            self.env.set_training_agent(0)
        self.env.seed(seed)
        self.action_space = self.env.action_space

    def get_observations(self):
        return self.env.get_observations()

    def get_rewards(self):
        if self.use_cython:
            return self.env.get_rewards()
        return self.env._get_rewards()

    def get_done(self):
        if self.use_cython:
            return self.env.get_done()
        return self.env._get_done()

    def step(self, actions):
        if self.use_cython:
            actions = np.asarray(actions).astype(np.uint8)
            self.env.step(actions)
            return self.get_observations(), self.get_rewards(), self.get_done()
        return self.env.step(actions)[:3]

    def act(self, state):
        if self.use_cython:
            return [opponent.act(state[i + 1], self.action_space) for i, opponent in enumerate(self.opponents)]
        return self.env.act(state)

    def get_game_state(self):
        if self.use_cython:
            return self.env.get_state()
        return self.env.get_json_info()

    def set_game_state(self, game_state):
        if self.use_cython:
            self.env.set_state(game_state)
        self.env._init_game_state = game_state
        self.env.set_json_info()

    def reset(self):
        obs = self.env.reset()
        if self.use_cython:
            obs = self.get_observations()
        return obs

    def get_features(self):
        if self.use_cython:
            features = self.env.get_features()
            features = features[0]
            return features
        state = self.get_observations()
        return extract_observation(state, 3, 800)
