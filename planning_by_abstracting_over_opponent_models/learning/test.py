from typing import List

import pommerman.agents
import torch

from planning_by_abstracting_over_opponent_models.config import cpu
from planning_by_abstracting_over_opponent_models.learning.pommerman_env_utils import create_agent_model
from planning_by_abstracting_over_opponent_models.learning.rl_agent import RLAgent
from planning_by_abstracting_over_opponent_models.pommerman_env.agents.modified_simple_agent import ModifiedSimpleAgent
from planning_by_abstracting_over_opponent_models.pommerman_env.pommerman_cython_env import PommermanCythonEnv

if __name__ == '__main__':
    device = cpu
    nb_opponents = 3
    opponent_class = ModifiedSimpleAgent
    iterations = int(9e4)
    agent_model = create_agent_model(0, 32, 6, nb_opponents, 3, 32, 64, 64, None, None, device, False)
    agent_model.load_state_dict(torch.load(f"../models/agent_model_{iterations}.pt"))
    agent_model.eval()
    agent = RLAgent(0, agent_model)
    agents: List[pommerman.agents.BaseAgent] = [opponent_class() for _ in range(nb_opponents)]
    agents.insert(0, agent)
    env = PommermanCythonEnv(agents, 1)
    action_space = env.action_space
    state = env.reset()
    done = False
    while not done:
        obs = env.get_features(state).to(device)
        agent_action = agent.act(obs, action_space)
        opponents_action = env.act(state)
        actions = [agent_action, *opponents_action]
        print(actions)
        # sleep(3)
        state, rewards, done = env.step(actions)
        env.render()
        # sleep(3)
