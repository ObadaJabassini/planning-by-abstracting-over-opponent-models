import argparse
import math
from multiprocessing import Pool, cpu_count
from random import randint

import numpy as np
import pommerman
import torch

from planning_by_abstracting_over_opponent_models.config import cpu
from planning_by_abstracting_over_opponent_models.learning.model.agent_model import create_agent_model
from planning_by_abstracting_over_opponent_models.learning.pommerman_env_utils import str_to_opponent_class
from planning_by_abstracting_over_opponent_models.planning.smmcts.smmcts import SMMCTS
from planning_by_abstracting_over_opponent_models.planning.state_evaluator.neural_network_state_evaluator import \
    NeuralNetworkStateEvaluator
from planning_by_abstracting_over_opponent_models.planning.state_evaluator.random_rollout_state_evaluator import \
    RandomRolloutStateEvaluator
from planning_by_abstracting_over_opponent_models.pommerman_env.pommerman_cython_env import PommermanCythonEnv
from planning_by_abstracting_over_opponent_models.pommerman_env.pommerman_python_env import PommermanPythonEnv


class DummyAgent(pommerman.agents.BaseAgent):

    def act(self, obs, action_space):
        pass


def play_game(game_id,
              play_id,
              seed,
              opponent_class,
              nb_players,
              use_cython,
              mcts_iterations,
              exploration_coef,
              fpu,
              ignore_opponent_actions,
              pw_c,
              pw_alpha,
              use_random_rollout):
    nb_actions = 6
    exploration_coefs = [exploration_coef] * nb_players
    fpus = [fpu] * nb_players
    random_players = [False] + ([ignore_opponent_actions] * (nb_players - 1))
    pw_cs = [pw_c] * nb_players
    pw_alphas = [pw_alpha] * nb_players
    if use_random_rollout:
        state_evaluator = RandomRolloutStateEvaluator(nb_players,
                                                      nb_actions,
                                                      pw_cs,
                                                      pw_alphas)
    else:
        iterations = int(9e4)
        agent_model = create_agent_model(0, 32, 6, nb_players - 1, 3, 32, 64, 64, None, None, cpu, False)
        agent_model.load_state_dict(torch.load(f"../saved_models/agent_model_{iterations}.pt"))
        agent_model.eval()
        state_evaluator = NeuralNetworkStateEvaluator(0, nb_actions, agent_model, agent_pw_c=1, agent_pw_alpha=1)
    smmcts = SMMCTS(nb_players=nb_players,
                    nb_actions=nb_actions,
                    exploration_coefs=exploration_coefs,
                    state_evaluator=state_evaluator)
    agents = [opponent_class() for _ in range(nb_players - 1)]
    agents.insert(0, DummyAgent())
    env = PommermanCythonEnv(agents=agents, seed=seed, rescale_rewards=True) if use_cython else \
        PommermanPythonEnv(agents=agents, seed=seed, rescale_rewards=True)
    state = env.reset()
    done = False
    frames = []
    while not done:
        actions = env.act(state)
        agent_action = smmcts.infer(env,
                                    iterations=mcts_iterations,
                                    fpus=fpus,
                                    random_players=random_players)
        actions.insert(0, agent_action)
        state, rewards, done = env.step(actions)
    win = int(rewards[0] == 1)
    tie = int(np.all(rewards == rewards[0]))
    return game_id, play_id, win, tie


parser = argparse.ArgumentParser()
parser.add_argument('--nb-processes', type=int, default=cpu_count() - 1)
parser.add_argument('--multiprocessing', dest="multiprocessing", action="store_true")
parser.add_argument('--no-multiprocessing', dest="multiprocessing", action="store_false")
parser.add_argument('--nb-games', type=int, default=1)
parser.add_argument('--nb-plays', type=int, default=1)
parser.add_argument('--nb-players', type=int, default=4, choices=[2, 4])
parser.add_argument('--opponent-class', type=str, default="simple")
parser.add_argument('--ignore-opponent-actions', dest="ignore_opponent_actions", action="store_true")
parser.add_argument('--search-opponent-actions', dest="ignore_opponent_actions", action="store_false")
parser.add_argument('--mcts-iterations', type=int, default=5000)
parser.add_argument('--exploration-coef', type=float, default=math.sqrt(2))
parser.add_argument('--fpu', type=float, default=0.25)
parser.add_argument('--pw-c', type=float, default=None)
parser.add_argument('--pw-alpha', type=float, default=None)
parser.add_argument('--use-random-rollout', dest="use_random_rollout", action="store_true")
parser.add_argument('--use-nn', dest="use_random_rollout", action="store_false")
parser.set_defaults(multiprocessing=True)
parser.set_defaults(ignore_opponent_actions=True)
parser.set_defaults(use_random_rollout=True)

if __name__ == '__main__':
    args = parser.parse_args()
    args.use_cython = args.nb_players == 4
    nb_games = args.nb_games
    nb_plays = args.nb_plays
    opponent_class_str = args.opponent_class
    opponent_class = str_to_opponent_class(opponent_class_str)
    games = []
    for game in range(1, nb_games + 1):
        seed = randint(0, int(1e6))
        for play in range(1, nb_plays + 1):
            params = (game,
                      play,
                      seed,
                      opponent_class,
                      args.nb_players,
                      args.use_cython,
                      args.mcts_iterations,
                      args.exploration_coef,
                      args.fpu,
                      args.ignore_opponent_actions,
                      args.pw_c,
                      args.pw_alpha,
                      args.use_random_rollout)
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
    s = f"opponent class = {opponent_class_str}, ignore = {args.ignore_opponent_actions}, fpu = {args.fpu}, win rate = {win_rate * 100}%, tie rate = {tie_rate * 100}%, lose rate = {lose_rate * 100}%"
    print(s)
