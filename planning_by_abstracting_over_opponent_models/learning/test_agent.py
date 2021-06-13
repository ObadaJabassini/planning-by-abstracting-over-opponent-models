import argparse
import os
import torch.multiprocessing as mp
from icecream import ic
from torch.multiprocessing import Pool, cpu_count
from random import randint
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from planning_by_abstracting_over_opponent_models.config import cpu
from planning_by_abstracting_over_opponent_models.learning.pommerman_env_utils import create_agent_model, \
    str_to_opponent_class
from planning_by_abstracting_over_opponent_models.pommerman_env.agents.pommerman_agent import PommermanAgent
from planning_by_abstracting_over_opponent_models.pommerman_env.agents.rl_agent import RLAgent
from planning_by_abstracting_over_opponent_models.pommerman_env.pommerman_cython_env import PommermanCythonEnv


def play_game(game_id,
              play_id,
              seed,
              agent_model,
              opponent_class,
              nb_opponents,
              device,
              render=False):
    agents: List[PommermanAgent] = [opponent_class() for _ in range(nb_opponents)]
    agent = RLAgent(0, agent_model)
    agents.insert(0, agent)
    env = PommermanCythonEnv(agents, seed)
    action_space = env.action_space
    state = env.reset()
    done = False
    while not done:
        obs = env.get_features(state).to(device)
        action_probs, _, _, _, opponent_influence = agent.estimate(obs)
        action_probs = F.softmax(action_probs, dim=-1).view(-1)
        agent_action = action_probs.argmax()
        agent_action = agent_action.item()
        opponents_action = env.act(state)
        actions = [agent_action, *opponents_action]
        state, rewards, done = env.step(actions)
        if render:
            env.render()
            ic(opponent_influence)
    win = int(rewards[0] == 1)
    tie = int(np.all(rewards == rewards[0]))
    print(f"game {game_id}, play {play_id} finished.")
    return game_id, play_id, win, tie


parser = argparse.ArgumentParser()
parser.add_argument('--nb-processes', type=int, default=cpu_count() - 1)
parser.add_argument('--multiprocessing', dest="multiprocessing", action="store_true")
parser.add_argument('--no-multiprocessing', dest="multiprocessing", action="store_false")
parser.add_argument('--nb-games', type=int, default=10)
parser.add_argument('--nb-plays', type=int, default=10)
parser.add_argument('--opponent-class', type=str, default="static")
parser.add_argument('--model-iteration', type=int, default=2580)
parser.add_argument('--rendering', dest="render", action="store_true")
parser.add_argument('--no-rendering', dest="render", action="store_false")
parser.set_defaults(multiprocessing=True)
parser.set_defaults(render=True)


if __name__ == '__main__':
    device = cpu
    os.environ['OMP_NUM_THREADS'] = '1'
    mp.set_start_method('spawn')
    args = parser.parse_args()
    nb_games = args.nb_games
    nb_plays = args.nb_plays
    render = args.render
    nb_opponents = 3
    opponent_class_str = args.opponent_class
    opponent_class = str_to_opponent_class(opponent_class_str)
    agent_model = create_agent_model(0, 32, 6, nb_opponents, 4, 32, 64, 64, 4, 64, device, False)
    agent_model.load_state_dict(torch.load(f"../saved_models/{opponent_class_str}/agent_model_{args.model_iteration}.pt"))
    agent_model.eval()
    agent_model.share_memory()

    if render:
        play_game(1,
                  1,
                  seed=randint(1, 10000),
                  agent_model=agent_model,
                  opponent_class=opponent_class,
                  nb_opponents=nb_opponents,
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
                          opponent_class,
                          nb_opponents,
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
        s = f"opponent class = {opponent_class_str}, win rate = {win_rate * 100}%, tie rate = {tie_rate * 100}%, lose rate = {lose_rate * 100}%"
        print(s)
