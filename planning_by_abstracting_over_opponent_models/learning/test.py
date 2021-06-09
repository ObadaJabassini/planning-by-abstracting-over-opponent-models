from typing import List

import pommerman.agents
import torch
import torch.nn.functional as F

from planning_by_abstracting_over_opponent_models.config import cpu
from planning_by_abstracting_over_opponent_models.learning.pommerman_env_utils import create_agent_model, \
    str_to_opponent_class
from planning_by_abstracting_over_opponent_models.pommerman_env.agents.pommerman_agent import PommermanAgent
from planning_by_abstracting_over_opponent_models.pommerman_env.agents.rl_agent import RLAgent
from planning_by_abstracting_over_opponent_models.pommerman_env.pommerman_cython_env import PommermanCythonEnv

if __name__ == '__main__':
    device = cpu
    nb_opponents = 3
    opponent_class = "static"
    opponent_class = str_to_opponent_class(opponent_class)
    iterations = int(9e4)
    agent_model = create_agent_model(0, 32, 6, nb_opponents, 4, 32, 64, 64, None, None, device, False)
    agent_model.load_state_dict(torch.load(f"../saved_models/agent_model.pt"))
    agent_model.eval()
    agent = RLAgent(0, agent_model)
    agents: List[PommermanAgent] = [opponent_class() for _ in range(nb_opponents)]
    agents.insert(0, agent)
    env = PommermanCythonEnv(agents, 1)
    action_space = env.action_space
    state = env.reset()
    done = False
    while not done:
        obs = env.get_features(state).to(device)
        agent_policy, agent_value, opponent_log_prob, opponent_value, _ = agent.estimate(obs)
        agent_probs = F.softmax(agent_policy, dim=-1).view(-1)
        agent_action = agent_probs.argmax().item()
        opponent_log_prob = F.softmax(opponent_log_prob.squeeze(0), dim=-1)
        print(agent_probs)
        opponents_action = env.act(state)
        actions = [agent_action, *opponents_action]
        print(actions)
        # sleep(3)
        state, rewards, done = env.step(actions)
        env.render()
        print(done)
        print(rewards)
        # sleep(3)
