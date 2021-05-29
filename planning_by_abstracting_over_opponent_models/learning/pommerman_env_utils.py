from typing import List

import pommerman
from pommerman.agents import RandomAgent

from planning_by_abstracting_over_opponent_models.learning.agent_model import create_agent_model
from planning_by_abstracting_over_opponent_models.learning.rl_agent import RLAgent
from planning_by_abstracting_over_opponent_models.pommerman_env.agents.cautious_agent import CautiousAgent
from planning_by_abstracting_over_opponent_models.pommerman_env.agents.modified_simple_agent import ModifiedSimpleAgent
from planning_by_abstracting_over_opponent_models.pommerman_env.agents.smart_random_agent import SmartRandomAgent, \
    SmartRandomAgentNoBomb
from planning_by_abstracting_over_opponent_models.pommerman_env.agents.static_agent import StaticAgent
from planning_by_abstracting_over_opponent_models.pommerman_env.pommerman_base_env import PommermanBaseEnv
from planning_by_abstracting_over_opponent_models.pommerman_env.pommerman_cython_env import PommermanCythonEnv
from planning_by_abstracting_over_opponent_models.pommerman_env.pommerman_python_env import PommermanPythonEnv


def str_to_opponent_class(s):
    return {
        "static": StaticAgent,
        "random": RandomAgent,
        "smart_no_bomb": SmartRandomAgentNoBomb,
        "smart": SmartRandomAgent,
        "simple": ModifiedSimpleAgent,
        "cautious": CautiousAgent
    }[s.lower()]


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
    opponent_class = str_to_opponent_class(opponent_class)
    agents: List[pommerman.agents.BaseAgent] = [opponent_class() for _ in range(nb_opponents)]
    agents.insert(0, agent)
    r = seed + rank
    env: PommermanBaseEnv = PommermanCythonEnv(agents, r) if use_cython else PommermanPythonEnv(agents, r)
    return agents, env
