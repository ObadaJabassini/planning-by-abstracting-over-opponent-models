import math
from random import randint
from typing import List

import numpy as np
import pommerman
import torch
from array2gif import write_gif
from tqdm import tqdm

from planning_by_abstracting_over_opponent_models.pommerman_env.base_pommerman_env import BasePommermanEnv
from planning_by_abstracting_over_opponent_models.planning.random_rollout_state_evaluator import \
    RandomRolloutStateEvaluator
from planning_by_abstracting_over_opponent_models.planning.state_evaluator import StateEvaluator
from planning_by_abstracting_over_opponent_models.planning.tree_node import TreeNode
from planning_by_abstracting_over_opponent_models.pommerman_env.pommerman_python_env import PommermanPythonEnv
from planning_by_abstracting_over_opponent_models.pommerman_env.pommerman_cython_env import PommermanCythonEnv


class SMMCTS:
    def __init__(self,
                 nb_players,
                 nb_actions,
                 exploration_coefs,
                 fpus,
                 state_evaluator: StateEvaluator):
        self.nb_players = nb_players
        self.nb_actions = nb_actions
        self.exploration_coefs = exploration_coefs
        self.fpus = fpus
        self.state_evaluator = state_evaluator

    def search(self, env, current_node: TreeNode):
        if current_node.is_terminal:
            return current_node.value_estimate
        # select
        actions = self.select(current_node)
        state, rewards, is_terminal = env.step(actions)
        rewards = rewards[:self.nb_players]
        # expand
        if actions not in current_node.children:
            value_estimate = self.expand(env, state, rewards, is_terminal, actions, current_node)
        else:
            child = current_node.children[actions]
            value_estimate = self.search(env, child)
        # backpropagate
        self.backpropagate(current_node, actions, value_estimate)
        return value_estimate

    def select(self, node):
        return node.best_actions()

    def expand(self, env, state, rewards, is_terminal, actions, current_node):
        if is_terminal:
            value_estimate = torch.as_tensor(rewards)
            action_prob_estimate = torch.full((self.nb_players, self.nb_actions), 1 / self.nb_actions)
            pw_alphas = [None] * self.nb_players
        else:
            value_estimate, action_prob_estimate, pw_alphas = self.state_evaluator.evaluate(env)
        current_node.children[actions] = TreeNode(state=state,
                                                  parent=current_node,
                                                  is_terminal=is_terminal,
                                                  value_estimate=value_estimate,
                                                  action_prob_estimate=action_prob_estimate,
                                                  nb_players=self.nb_players,
                                                  nb_actions=self.nb_actions,
                                                  exploration_coefs=self.exploration_coefs,
                                                  fpus=self.fpus,
                                                  pw_alphas=pw_alphas)
        return value_estimate

    def backpropagate(self, node, actions, value_estimate):
        node.update_actions_estimates(actions, value_estimate)

    def infer(self, env, iterations=100, progress_bar=False):
        initial_state = env.get_observations()
        value_estimate, action_probs_estimate, pw_alphas = self.state_evaluator.evaluate(env)
        root = TreeNode(initial_state,
                        None,
                        False,
                        value_estimate,
                        action_probs_estimate,
                        self.nb_players,
                        self.nb_actions,
                        self.exploration_coefs,
                        pw_alphas)
        game_state = env.get_game_state()
        r = range(iterations)
        if progress_bar:
            r = tqdm(r)
        for _ in r:
            self.search(env, root)
            env.set_game_state(game_state)
        most_visited_actions = root.most_visited_actions()
        most_visited_action = most_visited_actions[0]
        return most_visited_action


def heuristic_evaluator(initial_state, state):
    result = []
    # similar for all the agents
    s, d = initial_state[0], state[0]
    base_value = 0.17 * (len(s["alive"]) - len(d["alive"]))
    s_nb_wooden = (s["board"] == 2).sum()
    d_nb_wooden = (d["board"] == 2).sum()
    base_value += 0.1 * (s_nb_wooden - d_nb_wooden)
    for i in range(len(initial_state)):
        s, d = initial_state[i], state[i]
        value = base_value
        # (could be) different for each agent
        value += 0.15 * (s["blast_strength"] - d["blast_strength"])
        value += 0.15 * (int(s["can_kick"]) - int(d["can_kick"]))
        result.append(value)
    return result


if __name__ == '__main__':

    class DummyAgent(pommerman.agents.BaseAgent):

        def act(self, obs, action_space):
            pass


    move_map = {
        0: "Stop",
        1: "Up",
        2: "Down",
        3: "Left",
        4: "Right",
        5: "Bomb"
    }
    use_cython = False
    save_gif = False
    games = 10
    plays_per_game = 10
    opponent_class = pommerman.agents.SimpleAgent
    # 2 or 4
    nb_players = 2
    nb_actions = 6
    mcts_iterations = 100
    exploration_coefs = [math.sqrt(2)] * nb_players
    fpus = [1000] * nb_players
    # fpus = [1 / nb_players] * nb_players
    pw_alphas = [None] * nb_players
    depth = None
    heuristic_func = None
    # depth = 12
    # heuristic_func = heuristic_evaluator
    state_evaluator = RandomRolloutStateEvaluator(nb_players, nb_actions, pw_alphas, depth=depth, heuristic_func=heuristic_func)
    smmcts = SMMCTS(nb_players=nb_players,
                    nb_actions=nb_actions,
                    exploration_coefs=exploration_coefs,
                    fpus=fpus,
                    state_evaluator=state_evaluator)
    win_rate = 0
    tie_rate = 0
    progress_bar = False
    for game in range(1, games + 1):
        print(f"Game {game} started.")
        seed = randint(0, int(1e6))
        for play in range(1, plays_per_game + 1):
            print(f"Play {play} started.")
            agents: List[pommerman.agents.BaseAgent] = [opponent_class() for _ in range(nb_players - 1)]
            agents.insert(0, DummyAgent())
            env: BasePommermanEnv = PommermanCythonEnv(agents=agents, seed=seed) if use_cython else PommermanPythonEnv(agents=agents, seed=seed)
            state = env.reset()
            done = False
            step = 0
            frames = []
            while not done:
                step += 1
                actions = env.act(state)
                agent_action = smmcts.infer(env, iterations=mcts_iterations, progress_bar=progress_bar)
                actions.insert(0, agent_action)
                state, rewards, done = env.step(actions)
                # print(f"step {step}: {rewards}")
                if save_gif:
                    frame = env.render(mode="rgb_array")
                    frames.append(frame)
            rewards = np.asarray(rewards)
            win = int(rewards[0] == 1)
            tie = int(np.all(rewards == rewards[0]))
            win_rate += win
            tie_rate += tie
            if save_gif:
                file_name = f"games/game_{game}_play_{play}.gif"
                print("Saving gif..")
                write_gif(frames, file_name, 3)
    win_rate /= games * plays_per_game
    tie_rate /= games * plays_per_game
    lose_rate = 1 - win_rate - tie_rate
    print(f"win rate = {win_rate * 100}%, tie rate = {tie_rate * 100}%, lose rate = {lose_rate * 100}%")

