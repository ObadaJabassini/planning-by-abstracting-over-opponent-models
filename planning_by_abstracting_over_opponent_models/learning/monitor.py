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
            nb_actions,
            nb_opponents,
            opponent_class,
            save_interval,
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
    done = True
    start_time = time.time()
    # a quick hack to prevent the agent from stucking
    actions = deque(maxlen=1000)
    episode_length = 0
    episodes = 0
    with open("rewards.csv", "w") as f:
        f.write("Episode, Reward\n")
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
                reward = rewards[0]
                episodes += 1
                with open("rewards.csv", "a") as f:
                    f.write(f"{episodes}, {reward}\n")
                if episodes % save_interval == 0:
                    torch.save(shared_model.state_dict(), f"../saved_models/agent_model_{episodes}.pt")
                t = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time))
                t1 = counter.value / (time.time() - start_time)
                s = f"Episode: {episodes}, Time: {t}, num steps: {counter.value}, FPS: {t1:.0f}, episode reward: {reward}, episode length: {episode_length}"
                print(s)
                episode_length = 0
                actions.clear()
                state = env.reset()
                time.sleep(60)
    except KeyboardInterrupt:
        torch.save(shared_model.state_dict(), f"../saved_models/agent_model.pt")
