import argparse
import math
import os
import time

import torch.multiprocessing as mp
from icecream import ic
from torch.multiprocessing import Pool, cpu_count
from random import randint

import numpy as np
import pommerman
import torch

from planning_by_abstracting_over_opponent_models.config import cpu
from planning_by_abstracting_over_opponent_models.learning.model.agent_model import create_agent_model
from planning_by_abstracting_over_opponent_models.learning.pommerman_env_utils import str_to_agent
from planning_by_abstracting_over_opponent_models.planning.smmcts.smmcts import SMMCTS
from planning_by_abstracting_over_opponent_models.planning.smmcts.state_evaluator.neural_network_state_evaluator import \
    NeuralNetworkStateEvaluator
from planning_by_abstracting_over_opponent_models.planning.smmcts.state_evaluator.random_rollout_state_evaluator import \
    RandomRolloutStateEvaluator
from planning_by_abstracting_over_opponent_models.pommerman_env.agents.dummy_agent import DummyAgent
from planning_by_abstracting_over_opponent_models.pommerman_env.pommerman_cython_env import PommermanCythonEnv


def play_game(game_id,
              play_id,
              seed,
              opponent_classes,
              nb_players,
              nb_actions,
              exploration_coefs,
              fpus,
              random_players,
              state_evaluator,
              mcts_iterations):
    start_time = time.time()
    smmcts = SMMCTS(nb_players=nb_players,
                    nb_actions=nb_actions,
                    exploration_coefs=exploration_coefs,
                    fpus=fpus,
                    random_players=random_players,
                    state_evaluator=state_evaluator)
    agents = [opponent_class() for opponent_class in opponent_classes]
    agents.insert(0, DummyAgent())
    env = PommermanCythonEnv(agents=agents, seed=seed)
    state = env.reset()
    done = False
    while not done:
        actions = env.act(state)
        agent_action = smmcts.infer(env, iterations=mcts_iterations)
        actions.insert(0, agent_action)
        state, rewards, done = env.step(actions)
    win = int(rewards[0] == 1)
    tie = int(np.all(rewards == rewards[0]))
    elapsed_time = time.time() - start_time
    print(f"game id: {game_id}, play id: {play_id}, elapsed_time: {elapsed_time}")
    return game_id, play_id, win, tie


parser = argparse.ArgumentParser()
parser.add_argument('--nb-processes', type=int, default=cpu_count() - 1)
parser.add_argument('--multiprocessing', dest="multiprocessing", action="store_true")
parser.add_argument('--no-multiprocessing', dest="multiprocessing", action="store_false")
parser.add_argument('--nb-games', type=int, default=1)
parser.add_argument('--nb-plays', type=int, default=1)
parser.add_argument('--nb-players', type=int, default=4, choices=[2, 4])
ss = "simple, simple, simple"
parser.add_argument('--opponent-classes',
                    type=lambda sss: [str(item).strip().lower() for item in sss.split(',')],
                    default=ss)
parser.add_argument('--ignore-opponent-actions', dest="ignore_opponent_actions", action="store_true")
parser.add_argument('--search-opponent-actions', dest="ignore_opponent_actions", action="store_false")
parser.add_argument('--mcts-iterations', type=int, default=5000)
parser.add_argument('--model-iterations', type=int, default=1320)
parser.add_argument('--exploration-coef', type=float, default=math.sqrt(2))
parser.add_argument('--fpu', type=float, default=0.25)
parser.add_argument('--pw-c', type=float, default=None)
parser.add_argument('--pw-alpha', type=float, default=None)
parser.add_argument('--use-random-rollout', dest="use_random_rollout", action="store_true")
parser.add_argument('--use-nn', dest="use_random_rollout", action="store_false")
parser.set_defaults(multiprocessing=True)
parser.set_defaults(ignore_opponent_actions=False)
parser.set_defaults(use_random_rollout=False)

if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    mp.set_start_method('spawn')
    args = parser.parse_args()
    nb_actions = 6
    nb_players = args.nb_players
    nb_games = args.nb_games
    nb_plays = args.nb_plays
    mcts_iterations = args.mcts_iterations
    opponent_classes = args.opponent_classes
    combined_opponent_classes = ",".join(opponent_classes)
    opponent_classes = [str_to_agent(cl) for cl in opponent_classes]
    exploration_coefs = [args.exploration_coef] * nb_players
    fpus = [args.fpu] * nb_players
    random_players = [False] + ([args.ignore_opponent_actions] * (nb_players - 1))
    pw_cs = [args.pw_c] * nb_players
    pw_alphas = [args.pw_alpha] * nb_players
    if args.use_random_rollout:
        state_evaluator = RandomRolloutStateEvaluator(nb_players,
                                                      nb_actions,
                                                      pw_cs,
                                                      pw_alphas)
    else:
        agent_model = create_agent_model(rank=0,
                                         seed=randint(1, 1000),
                                         nb_actions=nb_actions,
                                         nb_opponents=nb_players - 1,
                                         nb_conv_layers=4,
                                         nb_filters=32,
                                         latent_dim=64,
                                         head_dim=64,
                                         nb_soft_attention_heads=4,
                                         hard_attention_rnn_hidden_size=64,
                                         approximate_hard_attention=True,
                                         attention_operation="add",
                                         device=cpu,
                                         train=False)
        agent_model.load_state_dict(torch.load(f"../../saved_models/{combined_opponent_classes}/agent_model_{args.model_iterations}.pt"))
        agent_model.eval()
        agent_model.share_memory()
        state_evaluator = NeuralNetworkStateEvaluator(0, nb_actions, agent_model, agent_pw_c=nb_actions, agent_pw_alpha=1)

    games = []
    for game_id in range(1, nb_games + 1):
        seed = randint(0, int(1e6))
        for play_id in range(1, nb_plays + 1):
            params = (game_id,
                      play_id,
                      seed,
                      opponent_classes,
                      nb_players,
                      nb_actions,
                      exploration_coefs,
                      fpus,
                      random_players,
                      state_evaluator,
                      mcts_iterations)
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
    s = f"opponent classes = {combined_opponent_classes}, ignore = {args.ignore_opponent_actions}, fpu = {args.fpu}, win rate = {win_rate * 100}%, tie rate = {tie_rate * 100}%, lose rate = {lose_rate * 100}%"
    print(s)
