# partially inspired from https://github.com/ikostrikov/pytorch-a3c/blob/master/train.py

import torch
import torch.nn.functional as F
import pommerman
from torch.optim import Adam

from planning_by_abstracting_over_opponent_models.config import gpu, cpu
from planning_by_abstracting_over_opponent_models.modeling.agent import Agent
from planning_by_abstracting_over_opponent_models.modeling.agent_loss import AgentLoss
from planning_by_abstracting_over_opponent_models.modeling.agent_model import AgentModel
from planning_by_abstracting_over_opponent_models.modeling.features_extractor import FeaturesExtractor


def get_board(state):
    board = state[0]['board']
    board = torch.FloatTensor(board)
    board = board.unsqueeze(0).unsqueeze(0)
    board = board.to(gpu)
    return board


def train():
    # pommerman
    features_extractor = FeaturesExtractor(image_size=11,
                                           conv_nb_layers=4,
                                           nb_filters=32,
                                           filter_size=3,
                                           filter_stride=1,
                                           filter_padding=1)
    features_extractor = features_extractor.to(gpu)
    action_space_size = 6
    nb_opponents = 1
    nb_units = 64
    agent_model = AgentModel(features_extractor=features_extractor,
                             nb_opponents=nb_opponents,
                             agent_nb_actions=action_space_size,
                             opponent_nb_actions=action_space_size,
                             nb_units=nb_units)
    agent_model = agent_model.to(gpu)
    agent = Agent(agent_model)
    agents = [
        agent,
        pommerman.agents.RandomAgent(),
    ]
    env = pommerman.make('PommeFFACompetition-v0', agents)
    action_space = env.action_space
    state = env.reset()
    # RL
    nb_episodes = 10000
    nb_steps = 10
    gamma = 0.9
    entropy_coef = 0.01
    value_coef = 0.5
    gae_lambda = 0.5
    value_loss_coef = 0.5
    opponent_coef = [0.1]
    optimizer = Adam(agent_model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)
    criterion = AgentLoss(gamma=gamma,
                          value_coef=value_coef,
                          entropy_coef=entropy_coef,
                          gae_lambda=gae_lambda,
                          value_loss_coef=value_loss_coef).to(gpu)

    for episode in range(nb_episodes):
        done = False
        agent_values = []
        agent_log_probs = []
        opponent_log_probs = []
        opponent_ground_truths = []
        agent_rewards = []
        agent_entropies = []
        for step in range(nb_steps):
            board = get_board(state)
            agent_policy, agent_value, opponent_policies = agent.estimate(board)
            agent_prob = F.softmax(agent_policy, dim=-1)
            agent_log_prob = F.log_softmax(agent_policy, dim=-1)
            opponent_log_prob = F.log_softmax(opponent_policies, dim=-1)
            agent_entropy = -(agent_log_prob * agent_prob).sum(1, keepdim=True)
            agent_action = agent_prob.multinomial(num_samples=1).detach()
            agent_log_prob = agent_log_prob.gather(1, agent_action)
            opponent_moves = [opponent.act(state, action_space) for opponent in agents[1:]]
            actions = [agent_action.item(), *opponent_moves]
            state, reward, done, info = env.step(actions)
            agent_rewards.append(reward)
            # only one step, no need for an additional dimension
            agent_entropies.append(agent_entropy.squeeze(0))
            agent_log_probs.append(agent_log_prob.squeeze(0))
            opponent_log_probs.append(opponent_log_prob.squeeze(0))
            agent_values.append(agent_value.squeeze(0))
            opponent_moves = torch.LongTensor(opponent_moves)
            opponent_ground_truths.append(opponent_moves)
            if done:
                state = env.reset()
                break
        R = torch.zeros(1, 1)
        if not done:
            board = get_board(state)
            _, agent_value, _ = agent.estimate(board)
            R = agent_value.detach()
        agent_values.append(R)
        optimizer.zero_grad()
        # transform everything
        # (nb_steps, nb_opponents, nb_actions)
        opponent_log_probs = torch.stack(opponent_log_probs)
        assert opponent_log_probs.shape == (nb_steps, nb_opponents, action_space_size)
        # (nb_opponents, nb_actions, nb_steps)
        opponent_log_probs = opponent_log_probs.permute(1, 2, 0)
        assert opponent_log_probs.shape == (nb_opponents, action_space_size, nb_steps)
        # (nb_steps, nb_opponents)
        opponent_ground_truths = torch.stack(opponent_ground_truths)
        assert opponent_ground_truths.shape == (nb_steps, nb_opponents)
        # (nb_opponents, nb_steps)
        opponent_ground_truths = opponent_ground_truths.permute(1, 0)
        assert opponent_ground_truths.shape == (nb_opponents, nb_steps)
        opponent_ground_truths = opponent_ground_truths.to(gpu)
        # backward step
        loss = criterion.forward(R,
                                 agent_rewards,
                                 agent_log_probs,
                                 agent_values,
                                 agent_entropies,
                                 opponent_log_probs,
                                 opponent_ground_truths,
                                 opponent_coef)
        print("arrived")
        # loss.backward()
    #     # optimization step
    #     optimizer.step()

    env.close()


if __name__ == '__main__':
    train()
