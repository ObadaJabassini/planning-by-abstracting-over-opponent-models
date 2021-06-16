import time
from collections import deque

import torch

from planning_by_abstracting_over_opponent_models.learning.pommerman_env_utils import create_env


def monitor(rank,
            seed,
            use_cython,
            shared_model,
            model_spec,
            nb_actions,
            nb_opponents,
            opponent_classes,
            save_interval,
            device):
    combined_opponent_classes = ",".join(opponent_classes)
    agents, env = create_env(rank,
                             seed,
                             use_cython,
                             model_spec,
                             nb_actions,
                             nb_opponents,
                             opponent_classes,
                             device,
                             train=False)
    agent = agents[0]
    agent_model = agent.agent_model
    action_space = env.action_space
    state = env.reset()
    done = True
    # a quick hack to prevent the agent from stucking
    actions = deque(maxlen=1000)
    episode_length = 0
    episodes = 0
    try:
        while True:
            episode_length += 1
            # Sync with the shared model
            if done:
                agent_model.load_state_dict(shared_model.state_dict())
            with torch.no_grad():
                obs = env.get_features(state).to(device)
                agent_action = agent.act(obs, action_space)
                opponents_action = env.act(state)
                episode_actions = [agent_action, *opponents_action]
                state, rewards, done = env.step(episode_actions)

            # a quick hack to prevent the agent from stucking
            actions.append(agent_action)
            if actions.count(actions[0]) == actions.maxlen:
                done = True
            if done:
                episodes += 1
                if episodes % save_interval == 0:
                    torch.save(shared_model.state_dict(), f"../saved_models/{combined_opponent_classes}/agent_model_{episodes}.pt")
                episode_length = 0
                actions.clear()
                state = env.reset()
                time.sleep(60)
    except:
        torch.save(shared_model.state_dict(), f"../saved_models/{combined_opponent_classes}/agent_model.pt")
