# partially inspired by https://github.com/ikostrikov/pytorch-a3c/blob/master/train.py

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

from planning_by_abstracting_over_opponent_models.env import create_env
from planning_by_abstracting_over_opponent_models.learning.agent_loss import AgentLoss


def collect_samples(env, state, lock, counter, agents, nb_opponents, nb_steps, device, dense_reward=True):
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
    agent = agents[0]
    while steps <= nb_steps:
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
        opponent_reward = torch.FloatTensor(opponent_reward).to(device)
        opponent_rewards.append(opponent_reward)
        opponent_log_probs.append(opponent_log_prob.squeeze(0))
        opponent_actions = torch.LongTensor(opponent_actions)
        opponent_actions_ground_truths.append(opponent_actions)
        opponent_values.append(opponent_value.view(-1))
        steps += 1
        with lock:
            counter.value += 1
        if done:
            episode_reward = agent_reward
            state = env.reset()
            break
    if not done:
        _, agent_value, _, opponent_value, _ = agent.estimate(state)
        r = agent_value.view(1)
        r = r.detach()
        opponent_value = opponent_value.view(-1)
    else:
        r = torch.zeros(1, device=device)
        opponent_value = torch.zeros(nb_opponents, device=device)
    r = r.to(device)
    opponent_value = opponent_value.to(device)
    agent_values.append(r)
    opponent_values.append(opponent_value)
    return steps, state, done, episode_reward, agent_rewards, agent_values, agent_log_probs, agent_entropies, \
           opponent_log_probs, opponent_actions_ground_truths, opponent_rewards, opponent_values


def prepare_tensors_for_loss_func(steps,
                                  nb_opponents,
                                  action_space_size,
                                  opponent_log_probs,
                                  opponent_actions_ground_truths,
                                  opponent_rewards,
                                  opponent_values,
                                  device):
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
    opponent_actions_ground_truths = opponent_actions_ground_truths.to(device)

    # (nb_steps, nb_opponents)
    opponent_rewards = torch.stack(opponent_rewards)
    assert opponent_rewards.shape == (steps, nb_opponents), f"{opponent_rewards.shape}"
    opponent_rewards = opponent_rewards.T
    # (nb_steps + 1, nb_opponents), not sure!
    opponent_values = torch.stack(opponent_values)
    assert opponent_values.shape == (steps + 1, nb_opponents), f"{opponent_values.shape}"
    opponent_values = opponent_values.T

    return opponent_log_probs, opponent_actions_ground_truths, opponent_rewards, opponent_values


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train(rank, seed, shared_model, counter, lock, device, action_space_size, nb_opponents, max_steps, optimizer=None):
    torch.manual_seed(seed + rank)
    agents, agent_model, env = create_env(seed, rank, device, action_space_size, nb_opponents, max_steps)
    state = env.reset()
    # RL
    dense_reward = True
    nb_episodes = 200
    nb_steps = 16
    max_grad_norm = 50
    gamma = 0.99
    entropy_coef = 0.01
    value_coef = 0.5
    gae_lambda = 1.0
    value_loss_coef = 0.5
    opponent_coefs = torch.tensor([0.1] * nb_opponents, device=device)
    if optimizer is None:
        optimizer = Adam(agent_model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)
    criterion = AgentLoss(gamma=gamma,
                          value_coef=value_coef,
                          entropy_coef=entropy_coef,
                          gae_lambda=gae_lambda,
                          value_loss_coef=value_loss_coef
                          ).to(device)
    episode = 1
    while episode <= nb_episodes:
        # sync with the shared model
        agent_model.load_state_dict(shared_model.state_dict())
        steps, state, done, episode_reward, agent_rewards, agent_values, agent_log_probs, agent_entropies, opponent_log_probs, \
        opponent_actions_ground_truths, opponent_rewards, opponent_values = collect_samples(env,
                                                                                            state,
                                                                                            lock,
                                                                                            counter,
                                                                                            agents,
                                                                                            nb_opponents,
                                                                                            nb_steps,
                                                                                            device,
                                                                                            dense_reward)
        opponent_log_probs, opponent_actions_ground_truths, opponent_rewards, opponent_values = prepare_tensors_for_loss_func(
            steps,
            nb_opponents,
            action_space_size,
            opponent_log_probs,
            opponent_actions_ground_truths,
            opponent_rewards,
            opponent_values,
            device
        )
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
        print(loss)
        # for name, param in agent_model.named_parameters():
        #     print(name, torch.isfinite(param).all())
        loss.backward()
        clip_grad_norm_(agent_model.parameters(), max_grad_norm)
        ensure_shared_grads(agent_model, shared_model)
        optimizer.step()
    env.close()
