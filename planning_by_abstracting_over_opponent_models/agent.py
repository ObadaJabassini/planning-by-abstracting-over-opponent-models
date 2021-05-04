import numpy as np
import torch
import torch.nn.functional as F
from pommerman.agents import BaseAgent


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


def get_observation(state, nb_opponents, max_steps, device):
    features = extract_features(state, nb_opponents, max_steps)
    # (18, 11, 11)
    obs = torch.from_numpy(features)
    # (18, 11, 11), swap width and height
    obs = obs.permute(0, 2, 1)
    obs = obs.float().unsqueeze(0).to(device)
    return obs


class Agent(BaseAgent):

    def __init__(self, agent_model, nb_opponents, max_steps, device, stochastic=False):
        super().__init__()
        self.agent_model = agent_model
        self.nb_opponents = nb_opponents
        self.max_steps = max_steps
        self.device = device
        self.stochastic = stochastic

    def act(self, obs, action_space):
        agent_policy, _, _, _, _ = self.estimate(obs)
        agent_prob = F.softmax(agent_policy, dim=-1)
        agent_prob = agent_prob.view(-1)
        agent_action = agent_prob.argmax() if not self.stochastic else agent_prob.multinomial(num_samples=1)
        agent_action = agent_action.item()
        return agent_action

    def estimate(self, obs):
        obs = get_observation(obs, self.nb_opponents, self.max_steps, self.device)
        return self.agent_model(obs)
