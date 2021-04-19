import time
from collections import deque

import torch

from planning_by_abstracting_over_opponent_models.env import create_env


def test(rank, seed, shared_model, counter, action_space_size, nb_opponents, max_steps, device):
    agents, agent_model, env = create_env(seed, rank, device, action_space_size, nb_opponents, max_steps, train=False)
    env.set_training_agent(0)
    state = env.reset()
    reward_sum = 0
    done = True
    start_time = time.time()
    # a quick hack to prevent the agent from stucking
    actions = deque(maxlen=100)
    episode_length = 0
    while True:
        episode_length += 1
        # Sync with the shared model
        if done:
            agent_model.load_state_dict(shared_model.state_dict())

        actions = env.act(state)
        with torch.no_grad():
            state, rewards, done, _ = env.step(actions)
        reward_sum += rewards[0]

        # a quick hack to prevent the agent from stucking
        actions.append(actions[0])
        if actions.count(actions[0]) == actions.maxlen:
            done = True
        if done:
            print("Time {}, num steps {}, FPS {:.0f}, episode reward {}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)),
                counter.value, counter.value / (time.time() - start_time),
                reward_sum, episode_length))
            reward_sum = 0
            episode_length = 0
            actions.clear()
            state = env.reset()
            time.sleep(60)

