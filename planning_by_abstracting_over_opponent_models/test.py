import pandas as pd
import altair as alt
import pommerman
from planning_by_abstracting_over_opponent_models.utils import get_board
from planning_by_abstracting_over_opponent_models.agent import Agent
from planning_by_abstracting_over_opponent_models.learning.agent_model import AgentModel


def load_agent_model():
    return AgentModel(None, None, None, None, None)


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
        done = False
        state = env.reset()
        episode_reward = 0
        steps = 0
        while not done:
            board = get_board(state)
            agent_action = agent.act(board, action_space)
            opponent_moves = [opponent.act(state, action_space) for opponent in agents[1:]]
            actions = [agent_action.item(), *opponent_moves]
            state, rewards, done, info = env.step(actions)
            agent_reward = rewards[0]
            episode_reward += agent_reward
            steps += 1
        episode_rewards.append(episode_reward)
    rewards_df = pd.DataFrame({"Episode": episode_range, "Reward": episode_rewards})
    chart = alt.Chart(rewards_df).mark_line().encode(
        x="Episode",
        y="Reward"
    )
    chart.save("total_rewards.png")


if __name__ == '__main__':
    test()
