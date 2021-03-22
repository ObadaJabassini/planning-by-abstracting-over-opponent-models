from typing import List

import pandas as pd
import altair as alt
import pommerman
import torch
from pommerman.agents import BaseAgent

from planning_by_abstracting_over_opponent_models.utils import get_board
from planning_by_abstracting_over_opponent_models.agent import Agent
from planning_by_abstracting_over_opponent_models.learning.agent_model import AgentModel


def load_agent_model():
    agent_model = torch.load("models/agent_model.model")
    agent_model.eval()
    return agent_model


def test():
    agent_model = load_agent_model()
    agent = Agent(agent_model)
    agent_index = 1
    nb_opponents = 1
    agents: List[BaseAgent] = [pommerman.agents.RandomAgent() for _ in range(nb_opponents)]
    agents.insert(agent_index, agent)
    opponent_agents = agents[:agent_index] + agents[agent_index + 1:]
    env = pommerman.make('PommeFFACompetition-v0', agents)
    action_space = env.action_space
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
            board = get_board(state, agent_index=agent_index)
            agent_action = agent.act(board, action_space)
            actions = [opponent.act(state, action_space) for opponent in opponent_agents]
            actions.insert(agent_index, agent_action)
            print(actions)
            state, rewards, done, info = env.step(actions)
            agent_reward = rewards[agent_index]
            episode_reward += agent_reward
        episode_rewards.append(episode_reward)
    rewards_df = pd.DataFrame({"Episode": episode_range, "Reward": episode_rewards})
    chart = alt.Chart(rewards_df).mark_line().encode(
        x="Episode",
        y="Reward"
    )
    chart.save("total_rewards.png")


if __name__ == '__main__':
    test()
