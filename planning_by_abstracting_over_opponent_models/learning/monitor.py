import time
from collections import deque

import torch

from planning_by_abstracting_over_opponent_models.learning.pommerman_env_utils import create_env


def monitor(rank,
            seed,
            use_cython,
            shared_model,
            counter,
            model_spec,
            nb_episodes,
            nb_actions,
            nb_opponents,
            opponent_class,
            device):
    agents, env = create_env(rank,
                             seed,
                             use_cython,
                             model_spec,
                             nb_actions,
                             nb_opponents,
                             opponent_class,
                             device,
                             train=False)
    agent = agents[0]
    agent_model = agent.agent_model
    action_space = env.action_space
    state = env.reset()
    reward_sum = 0
    done = True
    start_time = time.time()
    # a quick hack to prevent the agent from stucking
    actions = deque(maxlen=800)
    episode_length = 0
    episodes = 0
    while episodes < nb_episodes:
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
        reward_sum += rewards[0]

        # a quick hack to prevent the agent from stucking
        actions.append(agent_action)
        if actions.count(actions[0]) == actions.maxlen:
            done = True
        if done:
            episodes += 1
            t = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time))
            t1 = counter.value / (time.time() - start_time)
            print(
                f"Episode:{episodes}, Time {t}, num steps {counter.value}, FPS {t1:.0f}, episode reward {reward_sum}, episode length {episode_length}")
            reward_sum = 0
            episode_length = 0
            actions.clear()
            state = env.reset(raw_obs=False)
            # time.sleep(60)
