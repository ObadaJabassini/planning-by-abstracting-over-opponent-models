from typing import List

import pommerman

from planning_by_abstracting_over_opponent_models.learning.agent_model import create_agent_model
from planning_by_abstracting_over_opponent_models.learning.rl_agent import RLAgent
from planning_by_abstracting_over_opponent_models.pommerman_env.pommerman_base_env import PommermanBaseEnv
from planning_by_abstracting_over_opponent_models.pommerman_env.pommerman_cython_env import PommermanCythonEnv
from planning_by_abstracting_over_opponent_models.pommerman_env.pommerman_python_env import PommermanPythonEnv


def create_env(rank,
               seed,
               use_cython,
               model_spec,
               nb_actions,
               nb_opponents,
               opponent_class,
               device,
               train=True):
    agent_model = create_agent_model(rank,
                                     seed,
                                     nb_actions,
                                     nb_opponents,
                                     device=device,
                                     train=train,
                                     **model_spec)
    agent = RLAgent(0, agent_model)
    agents: List[pommerman.agents.BaseAgent] = [opponent_class() for _ in range(nb_opponents)]
    agents.insert(0, agent)
    r = seed + rank
    env: PommermanBaseEnv = PommermanCythonEnv(agents, r) if use_cython else PommermanPythonEnv(agents, r)
    return agents, env


