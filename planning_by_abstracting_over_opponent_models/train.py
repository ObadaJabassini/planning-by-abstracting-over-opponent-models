# partially inspired from https://github.com/ikostrikov/pytorch-a3c/blob/master/train.py
from typing import List

import altair as alt
import pandas as pd
import pommerman
import torch
import torch.nn.functional as F
from pommerman.agents import BaseAgent
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

from planning_by_abstracting_over_opponent_models.agent import Agent
from planning_by_abstracting_over_opponent_models.learning.agent_loss import AgentLoss
from planning_by_abstracting_over_opponent_models.learning.agent_model import AgentModel
from planning_by_abstracting_over_opponent_models.learning.features_extractor import FeaturesExtractor
from planning_by_abstracting_over_opponent_models.utils import gpu


def collect_samples(env, state, agent_index, agents, nb_opponents, nb_steps):
    agent_rewards = []
    agent_values = []
    agent_log_probs = []
    agent_entropies = []
    opponent_log_probs = []
    opponent_actions_ground_truths = []
    opponent_rewards = []
    opponent_values = []
    steps = 0
    nb_agents = len(agents)
    done = False
    episode_reward = 0
    agent = agents[agent_index]
    while steps <= nb_steps:
        agent_obs = state[agent_index]
        agent_policy, agent_value, opponent_policies, opponent_value, opponent_influence = agent.estimate(agent_obs)
        agent_prob = F.softmax(agent_policy, dim=-1)
        agent_log_prob = F.log_softmax(agent_policy, dim=-1)
        opponent_log_prob = F.log_softmax(opponent_policies, dim=-1)
        agent_entropy = -(agent_log_prob * agent_prob).sum(1, keepdim=True)
        agent_action = agent_prob.multinomial(num_samples=1).detach()
        agent_log_prob = agent_log_prob.gather(1, agent_action)

        actions = env.act(state)
        actions.insert(agent_index, agent_action.item())
        state, rewards, done, info = env.step(actions)
        # for a very strange reason, the env sometimes returns the wrong number of rewards
        rewards = rewards[:nb_agents]
        agent_reward = rewards[agent_index]
        opponent_reward = rewards[:agent_index] + rewards[agent_index + 1:]
        opponent_reward = torch.FloatTensor(opponent_reward).to(gpu)
        agent_rewards.append(agent_reward)
        opponent_rewards.append(opponent_reward)
        agent_entropies.append(agent_entropy.squeeze(0))
        agent_log_probs.append(agent_log_prob.view(-1))
        agent_values.append(agent_value.item())
        opponent_log_probs.append(opponent_log_prob.squeeze(0))
        opponent_moves = actions[:agent_index] + actions[agent_index + 1:]
        opponent_moves = torch.LongTensor(opponent_moves)
        opponent_actions_ground_truths.append(opponent_moves)
        opponent_values.append(opponent_value.view(-1))
        steps += 1
        if done:
            episode_reward = agent_reward
            state = env.reset()
            break
    R = torch.zeros(1, 1)
    opponent_value = torch.zeros(nb_opponents)
    if not done:
        agent_obs = state[agent_index]
        _, agent_value, _, opponent_value, _ = agent.estimate(agent_obs)
        R = agent_value.detach()
        opponent_value = opponent_value.view(-1)
    R = R.to(gpu)
    opponent_value = opponent_value.to(gpu)
    agent_values.append(R)
    opponent_values.append(opponent_value)
    return steps, state, done, R, episode_reward, agent_rewards, agent_values, agent_log_probs, agent_entropies, opponent_log_probs, opponent_actions_ground_truths, opponent_rewards, opponent_values


def prepare_tensors_for_loss_func(steps,
                                  nb_opponents,
                                  action_space_size,
                                  opponent_log_probs,
                                  opponent_actions_ground_truths,
                                  opponent_rewards,
                                  opponent_values):
    # (nb_steps, nb_opponents, nb_actions)
    opponent_log_probs = torch.stack(opponent_log_probs)
    assert opponent_log_probs.shape == (steps, nb_opponents, action_space_size), f"{opponent_log_probs.shape}"
    # (nb_opponents, nb_steps, nb_actions)
    opponent_log_probs = opponent_log_probs.permute(1, 0, 2)
    assert opponent_log_probs.shape == (nb_opponents, steps, action_space_size), f"{opponent_log_probs.shape}"
    # (nb_steps, nb_opponents)
    opponent_actions_ground_truths = torch.stack(opponent_actions_ground_truths)
    assert opponent_actions_ground_truths.shape == (steps, nb_opponents), f"{opponent_actions_ground_truths.shape}"
    # (nb_opponents, nb_steps)
    opponent_actions_ground_truths = opponent_actions_ground_truths.permute(1, 0)
    assert opponent_actions_ground_truths.shape == (nb_opponents, steps), f"{opponent_actions_ground_truths.shape}"
    opponent_actions_ground_truths = opponent_actions_ground_truths.to(gpu)

    # (nb_steps, nb_opponents)
    opponent_rewards = torch.stack(opponent_rewards)
    assert opponent_rewards.shape == (steps, nb_opponents), f"{opponent_rewards.shape}"
    opponent_rewards = opponent_rewards.T
    # (nb_steps + 1, nb_opponents), not sure!
    opponent_values = torch.stack(opponent_values)
    assert opponent_values.shape == (steps + 1, nb_opponents), f"{opponent_values.shape}"
    opponent_values = opponent_values.T

    return opponent_log_probs, opponent_actions_ground_truths, opponent_rewards, opponent_values


def train():
    # pommerman
    features_extractor = FeaturesExtractor(image_size=11,
                                           conv_nb_layers=4,
                                           nb_filters=32,
                                           filter_size=3,
                                           filter_stride=1,
                                           filter_padding=1)
    action_space_size = 6
    agent_index = 0
    nb_opponents = 1
    nb_units = 64
    agent_model = AgentModel(features_extractor=features_extractor,
                             nb_opponents=nb_opponents,
                             agent_nb_actions=action_space_size,
                             opponent_nb_actions=action_space_size,
                             nb_units=nb_units)
    agent_model = agent_model.to(gpu)
    agent = Agent(agent_model)
    agents: List[BaseAgent] = [pommerman.agents.SimpleAgent() for _ in range(nb_opponents)]
    agents.insert(agent_index, agent)
    env = pommerman.make('PommeFFACompetition-v0', agents)
    env.set_training_agent(agent_index)
    state = env.reset()
    # RL
    nb_episodes = 200
    nb_steps = 16
    max_grad_norm = 50
    gamma = 0.9
    entropy_coef = 0.01
    value_coef = 0.5
    gae_lambda = 1.0
    value_loss_coef = 0.5
    opponent_coefs = torch.tensor([0.1] * nb_opponents).to(gpu)
    optimizer = Adam(agent_model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)
    criterion = AgentLoss(gamma=gamma,
                          value_coef=value_coef,
                          entropy_coef=entropy_coef,
                          gae_lambda=gae_lambda,
                          value_loss_coef=value_loss_coef
                          ).to(gpu)
    episode = 1
    rewards = []
    losses = []
    running_loss = 0
    running_steps = 0
    nb_batches = 0
    while episode <= nb_episodes:
        steps, state, done, R, episode_reward, agent_rewards, agent_values, agent_log_probs, agent_entropies, opponent_log_probs, \
        opponent_actions_ground_truths, opponent_rewards, opponent_values = collect_samples(env,
                                                                                            state,
                                                                                            agent_index,
                                                                                            agents,
                                                                                            nb_opponents,
                                                                                            nb_steps)

        opponent_log_probs, opponent_actions_ground_truths, opponent_rewards, opponent_values = prepare_tensors_for_loss_func(
            steps,
            nb_opponents,
            action_space_size,
            opponent_log_probs,
            opponent_actions_ground_truths,
            opponent_rewards,
            opponent_values
        )
        # backward step
        optimizer.zero_grad()
        loss = criterion(R,
                         agent_rewards,
                         agent_log_probs,
                         agent_values,
                         agent_entropies,
                         opponent_log_probs,
                         opponent_actions_ground_truths,
                         opponent_values,
                         opponent_rewards,
                         opponent_coefs)
        loss.backward()
        clip_grad_norm_(agent_model.parameters(), max_grad_norm)
        optimizer.step()
        running_loss += loss.item()
        running_steps += steps
        nb_batches += 1
        if done:
            rewards.append(episode_reward)
            losses.append(running_loss // nb_batches)
            nb_batches = 0
            running_loss = 0
            running_steps = 0
            print(f"Episode {episode} finished.")
            episode += 1
    env.close()
    torch.save(agent_model, "models/agent_model.model")
    losses_df = pd.DataFrame({"Episode": range(nb_episodes), "Loss": losses})
    chart = alt.Chart(losses_df).mark_line().encode(
        x="Episode",
        y="Loss"
    )
    chart.save("figures/train_loss.png")


if __name__ == '__main__':
    train()
