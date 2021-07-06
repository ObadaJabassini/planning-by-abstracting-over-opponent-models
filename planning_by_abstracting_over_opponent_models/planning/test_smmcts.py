import argparse
import math
import os
import time

import torch.multiprocessing as mp
from torch.multiprocessing import Pool, cpu_count
from random import randint

import numpy as np
import torch

from planning_by_abstracting_over_opponent_models.config import cpu, gpu
from planning_by_abstracting_over_opponent_models.learning.model.agent_model import create_agent_model
from planning_by_abstracting_over_opponent_models.learning.pommerman_env_utils import str_to_agent
from planning_by_abstracting_over_opponent_models.planning.policy_estimator.neural_network_policy_estimator import \
    NeuralNetworkPolicyEstimator
from planning_by_abstracting_over_opponent_models.planning.policy_estimator.uniform_policy_estimator import \
    UniformPolicyEstimator
from planning_by_abstracting_over_opponent_models.planning.smmcts import SMMCTS
from planning_by_abstracting_over_opponent_models.planning.value_estimator.neural_network_value_estimator import \
    NeuralNetworkValueEstimator
from planning_by_abstracting_over_opponent_models.planning.value_estimator.random_rollout_value_estimator import \
    RandomRolloutValueEstimator
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
              value_estimator,
              policy_estimator,
              mcts_iterations):
    start_time = time.time()
    smmcts = SMMCTS(nb_players=nb_players,
                    nb_actions=nb_actions,
                    exploration_coefs=exploration_coefs,
                    fpus=fpus,
                    random_players=random_players,
                    value_estimator=value_estimator,
                    policy_estimator=policy_estimator)
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
parser.add_argument('--nb-games', type=int, default=2)
parser.add_argument('--nb-plays', type=int, default=2)
parser.add_argument('--nb-players', type=int, default=4, choices=[2, 4])
ss = "simple, simple, simple"
parser.add_argument('--opponent-classes',
                    type=lambda sss: [str(item).strip().lower() for item in sss.split(',')],
                    default=ss)
parser.add_argument('--ignore-opponent-actions', dest="ignore_opponent_actions", action="store_true")
parser.add_argument('--search-opponent-actions', dest="ignore_opponent_actions", action="store_false")
parser.add_argument('--mcts-iterations', type=int, default=10)
parser.add_argument('--model-iterations', type=int, default=27)
parser.add_argument('--exploration-coef', type=float, default=math.sqrt(2))
parser.add_argument('--fpu', type=float, default=0.25)
parser.add_argument('--pw-c', type=float, default=None)
parser.add_argument('--pw-alpha', type=float, default=None)
parser.add_argument('--policy-estimation', type=str, default="neural_network", choices=["uniform", "neural_network"])
parser.set_defaults(multiprocessing=True, ignore_opponent_actions=False)

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
    agent_model = create_agent_model(rank=0,
                                     seed=randint(1, 1000),
                                     nb_actions=nb_actions,
                                     nb_opponents=nb_players - 1,
                                     nb_conv_layers=4,
                                     nb_filters=32,
                                     latent_dim=128,
                                     nb_soft_attention_heads=4,
                                     hard_attention_rnn_hidden_size=None,
                                     approximate_hard_attention=True,
                                     device=cpu,
                                     train=False)
    f = f"../saved_models/{combined_opponent_classes}/agent_model_{args.model_iterations}.pt"
    agent_model.load_state_dict(torch.load(f))
    agent_model.eval()
    agent_model.share_memory()
    value_estimator = RandomRolloutValueEstimator(nb_players=nb_players, nb_actions=nb_actions)
    if args.policy_estimation == "uniform":
        policy_estimator = UniformPolicyEstimator(nb_players=nb_players,
                                                  nb_actions=nb_actions,
                                                  pw_cs=pw_cs,
                                                  pw_alphas=pw_alphas)
    else:
        policy_estimator = NeuralNetworkPolicyEstimator(agent_id=0,
                                                        agent_model=agent_model,
                                                        nb_actions=nb_actions)
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
                      value_estimator,
                      policy_estimator,
                      mcts_iterations)
            games.append(params)
    print("Started evaluating..")
    if args.multiprocessing:
        with Pool(args.nb_processes) as pool:
            result = pool.starmap(play_game, games)
    else:
        result = [play_game(*game) for game in games]
    wins = 0
    ties = 0
    for r in result:
        wins += r[2]
        ties += r[3]
    total_games = nb_games * nb_plays
    losses = total_games - wins - ties
    win_rate = wins / total_games
    tie_rate = ties / total_games
    lose_rate = losses / total_games
    s1 = f"opponent classes = {combined_opponent_classes}, ignore = {args.ignore_opponent_actions}, fpu = {args.fpu}"
    s2 = f"wins = {wins}, ties = {ties}, losses = {losses}"
    s3 = f"win rate = {win_rate * 100}%, tie rate = {tie_rate * 100}%, lose rate = {lose_rate * 100}%"
    print(s1)
    print(s2)
    print(s3)
