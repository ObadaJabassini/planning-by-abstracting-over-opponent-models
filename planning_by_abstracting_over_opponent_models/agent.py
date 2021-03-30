import pommerman
import numpy as np
import torch
import torch.nn.functional as F

from planning_by_abstracting_over_opponent_models.config import gpu


def extract_features(state, obs_width=17):
    board = state['board'].copy()
    agents = np.column_stack(np.where(board > 10))

    for i, agent in enumerate(agents):
        agent_id = board[agent[0], agent[1]]
        if agent_id not in state['alive']:  # < this fixes a bug >
            board[agent[0], agent[1]] = 0
        else:
            board[agent[0], agent[1]] = 11

    obs_radius = obs_width // 2
    pos = np.asarray(state['position'])

    # board
    board_pad = np.pad(board, (obs_radius, obs_radius), 'constant', constant_values=1)
    board_cent = board_cent = board_pad[pos[0]:pos[0] + 2 * obs_radius + 1, pos[1]:pos[1] + 2 * obs_radius + 1]

    # bomb blast strength
    bbs = state['bomb_blast_strength']
    bbs_pad = np.pad(bbs, (obs_radius, obs_radius), 'constant', constant_values=0)
    bbs_cent = bbs_cent = bbs_pad[pos[0]:pos[0] + 2 * obs_radius + 1, pos[1]:pos[1] + 2 * obs_radius + 1]

    # bomb life
    bl = state['bomb_life']
    bl_pad = np.pad(bl, (obs_radius, obs_radius), 'constant', constant_values=0)
    bl_cent = bl_cent = bl_pad[pos[0]:pos[0] + 2 * obs_radius + 1, pos[1]:pos[1] + 2 * obs_radius + 1]

    return np.concatenate((
        board_cent, bbs_cent, bl_cent,
        state['blast_strength'], state['can_kick'], state['ammo']), axis=None)


def get_observation(state, transform=False):
    features = extract_features(state) if transform else state["board"]
    obs = torch.FloatTensor(features)
    obs = obs.float().unsqueeze(0).unsqueeze(0)
    obs = obs.to(gpu)
    return obs


class Agent(pommerman.agents.BaseAgent):

    def __init__(self, agent_model):
        super().__init__()
        self.agent_model = agent_model

    def act(self, obs, action_space):
        agent_policy, _, _, _, _ = self.estimate(obs)
        agent_prob = F.softmax(agent_policy, dim=-1)
        agent_prob = agent_prob.view(-1)
        agent_action = agent_prob.argmax().item()
        # agent_action = agent_prob.multinomial(num_samples=1).detach()
        # agent_action = agent_action.item()
        return agent_action

    def estimate(self, obs):
        obs = get_observation(obs)
        return self.agent_model(obs)
