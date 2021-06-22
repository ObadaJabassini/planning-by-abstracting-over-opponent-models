from planning_by_abstracting_over_opponent_models.learning.model.agent_model import create_agent_model
from planning_by_abstracting_over_opponent_models.pommerman_env.agents.cautious_agent import CautiousAgent
from planning_by_abstracting_over_opponent_models.pommerman_env.agents.simple_agent import SimpleAgent
from planning_by_abstracting_over_opponent_models.pommerman_env.agents.random_agent import RandomAgent
from planning_by_abstracting_over_opponent_models.pommerman_env.agents.rl_agent import RLAgent
from planning_by_abstracting_over_opponent_models.pommerman_env.agents.smart_random_agent import SmartRandomAgent, \
    SmartRandomAgentNoBomb
from planning_by_abstracting_over_opponent_models.pommerman_env.agents.static_agent import StaticAgent
from planning_by_abstracting_over_opponent_models.pommerman_env.pommerman_base_env import PommermanBaseEnv
from planning_by_abstracting_over_opponent_models.pommerman_env.pommerman_cython_env import PommermanCythonEnv
from planning_by_abstracting_over_opponent_models.pommerman_env.pommerman_python_env import PommermanPythonEnv


def str_to_agent(s):
    d = {
        "static": StaticAgent,
        "random": RandomAgent,
        "smart_no_bomb": SmartRandomAgentNoBomb,
        "smart": SmartRandomAgent,
        "simple": SimpleAgent,
        "cautious": CautiousAgent
    }
    return d[s.strip().lower()]


def create_env(rank,
               seed,
               use_cython,
               model_spec,
               nb_actions,
               nb_opponents,
               opponent_classes,
               device,
               train=True):
    agent_model = create_agent_model(rank=rank,
                                     seed=seed,
                                     nb_actions=nb_actions,
                                     nb_opponents=nb_opponents,
                                     device=device,
                                     train=train,
                                     **model_spec)
    agent = RLAgent(0, agent_model)
    agents = [str_to_agent(opponent_class)() for opponent_class in opponent_classes]
    agents.insert(0, agent)
    r = seed + rank
    env: PommermanBaseEnv = PommermanCythonEnv(agents, r) if use_cython else PommermanPythonEnv(agents, r)
    return agents, env
