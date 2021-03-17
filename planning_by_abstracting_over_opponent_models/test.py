import pandas as pd
import altair as alt
import pommerman
import torch

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
    agents = [
        agent,
        pommerman.agents.RandomAgent(),
    ]
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
            board = get_board(state)
            agent_action = agent.act(board, action_space)
            opponent_moves = [opponent.act(state, action_space) for opponent in agents[1:]]
            actions = [agent_action, *opponent_moves]
            state, rewards, done, info = env.step(actions)
            print(rewards)
            agent_reward = rewards[0]
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
