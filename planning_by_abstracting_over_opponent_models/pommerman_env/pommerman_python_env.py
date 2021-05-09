import numpy as np
import pommerman
import torch

from planning_by_abstracting_over_opponent_models.pommerman_env.base_pommerman_env import BasePommermanEnv


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
    current_step_map = np.full(board_tuple, agent_state["step_count"] / max_steps)
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


class PommermanPythonEnv(BasePommermanEnv):

    def __init__(self, agents, seed):
        self.env = pommerman.make('PommeFFACompetition-v0', agents)
        self.env.seed(seed)
        self.env.set_training_agent(0)
        self.action_space = self.env.action_space

    def get_observations(self):
        return self.env.get_observations()

    def get_done(self):
        return self.env._get_done()

    def get_rewards(self):
        return self.env._get_rewards()

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        return self.env.step(actions)[:3]

    def act(self, state):
        return self.env.act(state)

    def render(self, mode=None):
        if mode is None:
            mode = 'human'
        return self.env.render(mode)

    def get_game_state(self):
        return self.env.get_json_info()

    def set_game_state(self, game_state):
        self.env._init_game_state = game_state
        self.env.set_json_info()

    def get_features(self):
        state = self.get_observations()
        return extract_observation(state, 3, 800)
