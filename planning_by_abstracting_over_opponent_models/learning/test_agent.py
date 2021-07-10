import pandas as pd
import altair as alt
import argparse
import os
from random import randint
from time import sleep

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from icecream import ic
from torch.multiprocessing import Pool, cpu_count

from planning_by_abstracting_over_opponent_models.config import cpu
from planning_by_abstracting_over_opponent_models.learning.pommerman_env_utils import create_agent_model, \
    str_to_agent
from planning_by_abstracting_over_opponent_models.pommerman_env.agents.rl_agent import RLAgent
from planning_by_abstracting_over_opponent_models.pommerman_env.pommerman_cython_env import PommermanCythonEnv


def play_game(game_id,
              play_id,
              seed,
              agent_model,
              opponent_classes,
              device,
              render=False):
    agents = [opponent_class() for opponent_class in opponent_classes]
    agent = RLAgent(0, agent_model)
    agents.insert(0, agent)
    env = PommermanCythonEnv(agents, seed)
    action_space = env.action_space
    state = env.reset()
    rewards = [0] * (nb_opponents + 1)
    done = False
    result = []
    while not done:
        obs = env.get_features(state).to(device)
        action_probs, _, opponent_log_prob, opponent_influence = agent.estimate(obs)
        action_probs = F.softmax(action_probs, dim=-1).view(-1)
        agent_action = action_probs.argmax()
        agent_action = agent_action.item()
        opponent_log_prob = opponent_log_prob.view(nb_opponents, -1)
        opponent_log_prob = F.softmax(opponent_log_prob, dim=-1)
        opponents_action = env.act(state)
        actions = [agent_action, *opponents_action]
        if render:
            ic(opponent_log_prob)
            ic(opponent_influence)
            # result.append(opponent_influence.squeeze(0).tolist())
        state, rewards, done = env.step(actions)
        if render:
            # sleep(0.3)
            env.render()
    win = int(rewards[0] == 1)
    tie = int(np.all(rewards == rewards[0]))
    print(f"game {game_id}, play {play_id} finished.")
    # df = pd.DataFrame.from_records(result, columns=["Agent", "Opponent 1", "Opponent 2", "Opponent 3"])

    return game_id, play_id, win, tie


parser = argparse.ArgumentParser()
parser.add_argument('--nb-processes', type=int, default=cpu_count() - 1)
parser.add_argument('--multiprocessing', dest="multiprocessing", action="store_true")
parser.add_argument('--no-multiprocessing', dest="multiprocessing", action="store_false")
parser.add_argument('--nb-games', type=int, default=10)
parser.add_argument('--nb-plays', type=int, default=10)
ss = "static, static, static"
parser.add_argument('--opponent-classes',
                    type=lambda sss: [str(item).strip().lower() for item in sss.split(',')],
                    default=ss)
parser.add_argument('--model-iteration', type=int, default=10)
parser.add_argument('--rendering', dest="render", action="store_true")
parser.add_argument('--no-rendering', dest="render", action="store_false")
parser.set_defaults(multiprocessing=True, render=True)


if __name__ == '__main__':
    device = cpu
    os.environ['OMP_NUM_THREADS'] = '1'
    mp.set_start_method('spawn')
    args = parser.parse_args()
    nb_games = args.nb_games
    nb_plays = args.nb_plays
    render = args.render
    nb_opponents = 3
    opponent_classes = args.opponent_classes
    combined_opponent_classes = ",".join(opponent_classes)
    opponent_classes = [str_to_agent(oc) for oc in opponent_classes]
    agent_model = create_agent_model(rank=1,
                                     seed=randint(1, 1000),
                                     nb_actions=6,
                                     nb_opponents=nb_opponents,
                                     nb_conv_layers=4,
                                     nb_filters=32,
                                     latent_dim=128,
                                     nb_soft_attention_heads=None,
                                     hard_attention_rnn_hidden_size=None,
                                     approximate_hard_attention=True,
                                     device=device,
                                     train=False)
    agent_model.load_state_dict(torch.load(f"../saved_models/{combined_opponent_classes}/agent_model_{args.model_iteration}.pt"))
    agent_model.eval()
    agent_model.share_memory()

    if render:
        play_game(game_id=1,
                  play_id=1,
                  seed=randint(1, 10000),
                  agent_model=agent_model,
                  opponent_classes=opponent_classes,
                  device=device,
                  render=True)
    else:
        games = []
        for game_id in range(1, nb_games + 1):
            seed = randint(0, int(1e6))
            for play_id in range(1, nb_plays + 1):
                params = (game_id,
                          play_id,
                          seed,
                          agent_model,
                          opponent_classes,
                          device,
                          False)
                games.append(params)

        if args.multiprocessing:
            with Pool(args.nb_processes) as pool:
                result = pool.starmap(play_game, games)
        else:
            result = [play_game(*game) for game in games]
        win_rate = 0
        tie_rate = 0
        for r in result:
            win_rate += r[2]
            tie_rate += r[3]
        total_games = nb_games * nb_plays
        win_rate /= total_games
        tie_rate /= total_games
        lose_rate = 1 - win_rate - tie_rate
        s = f"opponent classes = {combined_opponent_classes}, win rate = {win_rate * 100}%, tie rate = {tie_rate * 100}%, lose rate = {lose_rate * 100}%"
        print(s)
