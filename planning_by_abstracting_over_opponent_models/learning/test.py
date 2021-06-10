from time import sleep
from typing import List

import torch
import torch.nn.functional as F
from icecream import ic

from planning_by_abstracting_over_opponent_models.config import cpu
from planning_by_abstracting_over_opponent_models.learning.pommerman_env_utils import create_agent_model, \
    str_to_opponent_class
from planning_by_abstracting_over_opponent_models.pommerman_env.agents.pommerman_agent import PommermanAgent
from planning_by_abstracting_over_opponent_models.pommerman_env.agents.rl_agent import RLAgent
from planning_by_abstracting_over_opponent_models.pommerman_env.pommerman_cython_env import PommermanCythonEnv

if __name__ == '__main__':
    device = cpu
    nb_opponents = 3
    opponent_class_str = "static"
    opponent_class = str_to_opponent_class(opponent_class_str)
    iterations = int(9e4)
    agent_model = create_agent_model(0, 32, 6, nb_opponents, 4, 32, 64, 64, 4, 64, device, False)
    agent_model.load_state_dict(torch.load(f"../saved_models/agent_model_{opponent_class_str}_1500.pt"))
    agent_model.eval()
    agent = RLAgent(0, agent_model)
    agents: List[PommermanAgent] = [opponent_class() for _ in range(nb_opponents)]
    agents.insert(0, agent)
    env = PommermanCythonEnv(agents, 5)
    action_space = env.action_space
    state = env.reset()
    done = False
    while not done:
        obs = env.get_features(state).to(device)
        agent_policy, agent_value, opponent_log_prob, opponent_value, opponent_influence = agent.estimate(obs)
        agent_probs = F.softmax(agent_policy, dim=-1).view(-1)
        agent_action = agent_probs.argmax().item()
        opponent_probs = F.softmax(opponent_log_prob.squeeze(0), dim=-1)
        ic(agent_probs)
        ic(opponent_probs)
        ic(opponent_influence)
        opponents_action = env.act(state)
        actions = [agent_action, *opponents_action]
        ic(actions)
        # sleep(3)
        state, rewards, done = env.step(actions)
        env.render()
        # sleep(3)
