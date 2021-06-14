# partially inspired by https://github.com/ikostrikov/pytorch-a3c/blob/master/train.py
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from planning_by_abstracting_over_opponent_models.learning.model.agent_loss import AgentLoss
from planning_by_abstracting_over_opponent_models.learning.pommerman_env_utils import create_env
from planning_by_abstracting_over_opponent_models.learning.reward_shaping.reward_shaper import strs_to_reward_shaper

torch.autograd.set_detect_anomaly(True)


def reshape_tensors_for_loss_func(steps,
                                  nb_opponents,
                                  nb_actions,
                                  opponent_log_probs,
                                  opponent_actions_ground_truths,
                                  opponent_rewards,
                                  opponent_values,
                                  device):
    # (nb_steps, nb_opponents, nb_actions)
    opponent_log_probs = torch.stack(opponent_log_probs)
    assert opponent_log_probs.shape == (
        steps, nb_opponents,
        nb_actions), f"{opponent_log_probs.shape} != {(steps, nb_opponents, nb_actions)}"
    # (nb_opponents, nb_steps, nb_actions)
    opponent_log_probs = opponent_log_probs.permute(1, 0, 2)
    assert opponent_log_probs.shape == (
        nb_opponents, steps,
        nb_actions), f"{opponent_log_probs.shape} != {(nb_opponents, steps, nb_actions)}"
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


def collect_trajectory(env,
                       state,
                       lock,
                       counter,
                       agents,
                       nb_opponents,
                       nb_actions,
                       nb_steps,
                       device,
                       reward_shaper=None):
    agent_rewards = []
    agent_values = []
    agent_log_probs = []
    agent_entropies = []
    opponent_log_probs = []
    opponent_actions_ground_truths = []
    opponent_rewards = []
    opponent_values = []
    agent = agents[0]
    done = False
    steps = 0
    while not done and steps < nb_steps:
        steps += 1
        obs = env.get_features(state).to(device)
        agent_policy, agent_value, opponent_log_prob, opponent_value, opponent_influence = agent.estimate(obs)
        agent_prob = F.softmax(agent_policy, dim=-1)
        agent_log_prob = F.log_softmax(agent_policy, dim=-1)
        agent_entropy = -(agent_log_prob * agent_prob).sum(1, keepdim=True)
        agent_action = agent_prob.multinomial(num_samples=1).detach()
        agent_log_prob = agent_log_prob.gather(1, agent_action)
        opponent_actions = env.act(state)
        agent_action = agent_action.item()
        actions = [agent_action, *opponent_actions]
        state, rewards, done = env.step(actions)
        agent_reward = rewards[0]
        if reward_shaper is not None and not done and agent_reward == 0:
            agent_reward = reward_shaper.shape(state[0], agent_action)
        # agent
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
        reward_shaper.reset()
        r = torch.zeros(1, device=device)
        opponent_value = torch.zeros(nb_opponents, device=device)
    else:
        obs = env.get_features(state).to(device)
        _, agent_value, _, opponent_value, _ = agent.estimate(obs)
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
                                                        nb_actions,
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
          use_cython,
          shared_model,
          optimizer,
          counter,
          lock,
          model_spec,
          nb_steps,
          nb_actions,
          nb_opponents,
          opponent_classes,
          reward_shapers,
          max_grad_norm,
          device):
    combined_reward_shapers = ",".join(reward_shapers)
    combined_opponent_classes = ",".join(opponent_classes)
    agents, env = create_env(rank,
                             seed,
                             use_cython,
                             model_spec,
                             nb_actions,
                             nb_opponents,
                             opponent_classes,
                             device,
                             train=True)
    agent_model = agents[0].agent_model
    state = env.reset()
    # RL
    gamma = 0.99
    entropy_coef = 0.01
    value_loss_coef = 0.5
    gae_lambda = 1.0
    opponent_coefs = torch.tensor([0.01] * nb_opponents, device=device)
    if optimizer is None:
        optimizer = Adam(agent_model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)
    criterion = AgentLoss(gamma=gamma,
                          value_loss_coef=value_loss_coef,
                          entropy_coef=entropy_coef,
                          gae_lambda=gae_lambda).to(device)
    reward_shaper = strs_to_reward_shaper(reward_shapers)
    episodes = 0
    episode_batches = 0
    running_total_loss = 0.0
    running_cross_entropy_loss = 0.0
    p = f"runs/{combined_opponent_classes}/{combined_reward_shapers}"
    Path(p).mkdir(exist_ok=True, parents=True)
    summary_writer = SummaryWriter(p) if rank == 0 else None
    try:
        while True:
            # sync with the shared model
            agent_model.load_state_dict(shared_model.state_dict())
            steps, state, done, agent_trajectory, opponent_trajectory = collect_trajectory(env=env,
                                                                                           state=state,
                                                                                           lock=lock,
                                                                                           counter=counter,
                                                                                           agents=agents,
                                                                                           nb_opponents=nb_opponents,
                                                                                           nb_actions=nb_actions,
                                                                                           nb_steps=nb_steps,
                                                                                           device=device,
                                                                                           reward_shaper=reward_shaper)
            agent_rewards, agent_values, agent_log_probs, agent_entropies = agent_trajectory
            opponent_rewards, opponent_values, opponent_log_probs, opponent_actions_ground_truths = opponent_trajectory
            # backward step
            optimizer.zero_grad()
            total_loss, opponent_policy_loss, opponent_value_loss = criterion(agent_rewards,
                                                                              agent_log_probs,
                                                                              agent_values,
                                                                              agent_entropies,
                                                                              opponent_rewards,
                                                                              opponent_log_probs,
                                                                              opponent_values,
                                                                              opponent_actions_ground_truths,
                                                                              opponent_coefs)
            total_loss.backward()
            if max_grad_norm is not None:
                clip_grad_norm_(agent_model.parameters(), max_grad_norm)
            ensure_shared_grads(agent_model, shared_model)
            optimizer.step()

            running_total_loss += total_loss.item()
            running_cross_entropy_loss += opponent_policy_loss.item()
            episode_batches += 1

            if done:
                episodes += 1
                avg_loss = running_total_loss / episode_batches
                avg_cross_entropy_loss = running_cross_entropy_loss / episode_batches
                running_total_loss = 0.0
                running_cross_entropy_loss = 0.0
                episode_batches = 0
                if summary_writer is not None and episodes % 10 == 0:
                    summary_writer.add_scalar('training loss',
                                              avg_loss,
                                              episodes)
                    summary_writer.add_scalar('cross entropy loss',
                                              avg_cross_entropy_loss,
                                              episodes)
                    summary_writer.flush()
    except:
        if summary_writer is not None:
            summary_writer.close()
