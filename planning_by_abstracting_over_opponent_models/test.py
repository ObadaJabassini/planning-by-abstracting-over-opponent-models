from typing import List

import altair as alt
import pandas as pd
import pommerman
import torch
from pommerman.agents import BaseAgent

from planning_by_abstracting_over_opponent_models.agent import Agent


def load_agent_model():
    agent_model = torch.load("models/agent_model.model")
    agent_model.eval()
    return agent_model


def test():
    agent_index = 0
    nb_opponents = 1
    agent_model = load_agent_model()
    agent = Agent(agent_model)
    agents: List[BaseAgent] = [pommerman.agents.SimpleAgent() for _ in range(nb_opponents)]
    agents.insert(agent_index, agent)
    # print(agents)
    env = pommerman.make('PommeFFACompetition-v0', agents)
    # RL
    nb_episodes = 100
    episode_rewards = []
    episode_range = range(nb_episodes)
    for episode in episode_range:
        print(f"Episode {episode}")
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            env.render()
            actions = env.act(state)
            state, rewards, done, info = env.step(actions)
            episode_reward = rewards[agent_index]
        episode_rewards.append(episode_reward)
    rewards_df = pd.DataFrame({"Episode": episode_range, "Reward": episode_rewards})
    chart = alt.Chart(rewards_df).mark_line().encode(
        x="Episode",
        y="Reward"
    )
    chart.save("figures/test_rewards.png")


if __name__ == '__main__':
    test()
