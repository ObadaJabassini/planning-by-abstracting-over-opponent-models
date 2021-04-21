# partially inspired by https://github.com/ikostrikov/pytorch-a3c/blob/master/train.py

import torch
import torch.nn.functional as F
from icecream import ic
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

from planning_by_abstracting_over_opponent_models.env import create_env
from planning_by_abstracting_over_opponent_models.learning.agent_loss import AgentLoss


def reshape_tensors_for_loss_func(steps,
                                  nb_opponents,
                                  action_space_size,
                                  opponent_log_probs,
                                  opponent_actions_ground_truths,
                                  opponent_rewards,
                                  opponent_values,
                                  device):
    # (nb_steps, nb_opponents, nb_actions)
    opponent_log_probs = torch.stack(opponent_log_probs)
    assert opponent_log_probs.shape == (
        steps, nb_opponents,
        action_space_size), f"{opponent_log_probs.shape} != {(steps, nb_opponents, action_space_size)}"
    # (nb_opponents, nb_steps, nb_actions)
    opponent_log_probs = opponent_log_probs.permute(1, 0, 2)
    assert opponent_log_probs.shape == (
        nb_opponents, steps,
        action_space_size), f"{opponent_log_probs.shape} != {(nb_opponents, steps, action_space_size)}"
    # (nb_steps, nb_opponents)
    opponent_actions_ground_truths = torch.stack(opponent_actions_ground_truths).to(device)
    assert opponent_actions_ground_truths.shape == (
        steps, nb_opponents), f"{opponent_actions_ground_truths.shape} != {(steps, nb_opponents)}"
    # (nb_opponents, nb_steps)
    opponent_actions_ground_truths = opponent_actions_ground_truths.permute(1, 0)
    assert opponent_actions_ground_truths.shape == (
        nb_opponents, steps), f"{opponent_actions_ground_truths.shape} != {(nb_opponents, steps)}"
    opponent_actions_ground_truths = opponent_actions_ground_truths.to(device)

    # (nb_steps, nb_opponents)
    opponent_rewards = torch.stack(opponent_rewards).to(device)
    assert opponent_rewards.shape == (steps, nb_opponents), f"{opponent_rewards.shape} != {(steps, nb_opponents)}"
    opponent_rewards = opponent_rewards.T
    # (nb_steps + 1, nb_opponents), not sure!
    opponent_values = torch.stack(opponent_values)
    assert opponent_values.shape == (steps + 1, nb_opponents), f"{opponent_values.shape} != {(steps + 1, nb_opponents)}"
    opponent_values = opponent_values.T

    return opponent_rewards, opponent_values, opponent_log_probs, opponent_actions_ground_truths


def collect_trajectory(env, state, lock, counter, agents, nb_opponents, action_space_size, nb_steps, device,
                       dense_reward=True):
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
    agent = agents[0]
    while not done and steps < nb_steps:
        steps += 1
        agent_policy, agent_value, opponent_log_prob, opponent_value, opponent_influence = agent.estimate(state)
        agent_prob = F.softmax(agent_policy, dim=-1)
        agent_log_prob = F.log_softmax(agent_policy, dim=-1)
        agent_entropy = -(agent_log_prob * agent_prob).sum(1, keepdim=True)
        agent_action = agent_prob.multinomial(num_samples=1).detach()
        agent_log_prob = agent_log_prob.gather(1, agent_action)
        opponent_actions = env.act(state)
        agent_action = agent_action.item()
        actions = [agent_action, *opponent_actions]
        ammo_before = state[0]["ammo"]
        state, rewards, done, info = env.step(actions)
        ammo_after = state[0]["ammo"]
        # for a very strange reason, the env sometimes returns the wrong number of rewards
        rewards = rewards[:nb_agents]
        if dense_reward and rewards[0] == 0 and ammo_after > ammo_before:
            rewards[0] = 0.1
        # agent
        agent_reward = rewards[0]
        agent_rewards.append(agent_reward)
        agent_entropies.append(agent_entropy.view(-1))
        agent_log_probs.append(agent_log_prob.view(-1))
        agent_values.append(agent_value.view(-1))

        # opponents
        opponent_reward = rewards[1:]
        opponent_reward = torch.FloatTensor(opponent_reward)
        opponent_rewards.append(opponent_reward)
        opponent_log_probs.append(opponent_log_prob.squeeze(0))
        opponent_actions = torch.LongTensor(opponent_actions)
        opponent_actions_ground_truths.append(opponent_actions)
        opponent_values.append(opponent_value.view(-1))
        with lock:
            counter.value += 1
    if done:
        state = env.reset()
        r = torch.zeros(1, device=device)
        opponent_value = torch.zeros(nb_opponents, device=device)
    else:
        _, agent_value, _, opponent_value, _ = agent.estimate(state)
        r = agent_value.view(1)
        r = r.detach()
        opponent_value = opponent_value.view(-1)
    r = r.to(device)
    opponent_value = opponent_value.to(device)
    agent_values.append(r)
    opponent_values.append(opponent_value)

    agent_trajectory = (agent_rewards, agent_values, agent_log_probs, agent_entropies)
    opponent_trajectory = reshape_tensors_for_loss_func(steps,
                                                        nb_opponents,
                                                        action_space_size,
                                                        opponent_log_probs,
                                                        opponent_actions_ground_truths,
                                                        opponent_rewards,
                                                        opponent_values,
                                                        device)
    return steps, state, done, agent_trajectory, opponent_trajectory


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train(rank,
          seed,
          shared_model,
          counter,
          lock,
          model_spec,
          nb_episodes,
          action_space_size,
          nb_opponents,
          nb_steps,
          max_steps,
          device,
          optimizer):
    torch.manual_seed(seed + rank)
    agents, agent_model, env = create_env(seed,
                                          rank,
                                          device,
                                          model_spec,
                                          action_space_size,
                                          nb_opponents,
                                          max_steps,
                                          True)
    state = env.reset()
    # RL
    dense_reward = True
    max_grad_norm = 50
    gamma = 0.99
    entropy_coef = 0.01
    value_coef = 0.5
    gae_lambda = 1.0
    value_loss_coef = 0.5
    opponent_coefs = torch.tensor([0.01] * nb_opponents, device=device)
    if optimizer is None:
        optimizer = Adam(agent_model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)
    criterion = AgentLoss(gamma=gamma,
                          value_coef=value_coef,
                          entropy_coef=entropy_coef,
                          gae_lambda=gae_lambda,
                          value_loss_coef=value_loss_coef
                          ).to(device)
    episodes = 0
    while episodes < nb_episodes:
        # sync with the shared model
        agent_model.load_state_dict(shared_model.state_dict())
        steps, state, done, agent_trajectory, opponent_trajectory = collect_trajectory(env,
                                                                                       state,
                                                                                       lock,
                                                                                       counter,
                                                                                       agents,
                                                                                       nb_opponents,
                                                                                       action_space_size,
                                                                                       nb_steps,
                                                                                       device,
                                                                                       dense_reward)
        agent_rewards, agent_values, agent_log_probs, agent_entropies = agent_trajectory
        opponent_rewards, opponent_values, opponent_log_probs, opponent_actions_ground_truths = opponent_trajectory
        # ic(agent_rewards)
        # ic(agent_log_probs)
        # ic(agent_values)
        # ic(opponent_log_probs)
        # ic(opponent_actions_ground_truths)
        # ic(opponent_values)
        # backward step
        optimizer.zero_grad()
        loss = criterion(agent_rewards,
                         agent_log_probs,
                         agent_values,
                         agent_entropies,
                         opponent_log_probs,
                         opponent_actions_ground_truths,
                         opponent_values,
                         opponent_rewards,
                         opponent_coefs)
        # ic(loss)
        # for name, param in agent_model.named_parameters():
        #     print(name, torch.isfinite(param).all())
        loss.backward()
        clip_grad_norm_(agent_model.parameters(), max_grad_norm)
        ensure_shared_grads(agent_model, shared_model)
        optimizer.step()
        if done:
            episodes += 1
