import math
from random import randint
from typing import List

import pommerman
import torch
from tqdm import tqdm

from planning_by_abstracting_over_opponent_models.planning.random_rollout_state_evaluator import \
    RandomRolloutStateEvaluator
from planning_by_abstracting_over_opponent_models.planning.state_evaluator import StateEvaluator
from planning_by_abstracting_over_opponent_models.planning.tree_node import TreeNode
from planning_by_abstracting_over_opponent_models.pommerman_wrapped_env import PommermanWrappedEnv


class SMMCTS:
    def __init__(self,
                 nb_players,
                 nb_actions,
                 exploration_coefs,
                 state_evaluator: StateEvaluator,
                 use_progressive_widening=False):
        self.nb_players = nb_players
        self.nb_actions = nb_actions
        self.exploration_coefs = torch.as_tensor(exploration_coefs)
        self.state_evaluator = state_evaluator
        self.use_progressive_widening = use_progressive_widening

    def search(self, env, current_node: TreeNode):
        if current_node.is_terminal:
            return current_node.value_estimate
        # select
        actions = self.select(current_node)
        state, rewards, is_terminal = env.step(actions)
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
            value_estimate = torch.as_tensor(rewards[:self.nb_players])
            action_probs = torch.full((self.nb_players, self.nb_actions), 1 / self.nb_actions)
            opponent_influence = torch.full((self.nb_players - 1, ), 1 / (self.nb_players - 1))
        else:
            value_estimate, action_probs, opponent_influence = self.state_evaluator.evaluate(env)
        current_node.children[actions] = TreeNode(state=state,
                                                  parent=current_node,
                                                  is_terminal=is_terminal,
                                                  value_estimate=value_estimate,
                                                  action_prob_estimate=action_probs,
                                                  opponent_influence=opponent_influence,
                                                  nb_players=self.nb_players,
                                                  nb_actions=self.nb_actions,
                                                  exploration_coefs=self.exploration_coefs,
                                                  use_progressive_widening=self.use_progressive_widening)
        return value_estimate

    def backpropagate(self, node, actions, value_estimate):
        node.update_actions_estimates(actions, value_estimate)

    def infer(self, env, iterations=100, progress_bar=False):
        initial_state = env.get_observations()
        value_estimate, action_probs, opponent_influence = self.state_evaluator.evaluate(env)
        root = TreeNode(initial_state,
                        None,
                        False,
                        value_estimate,
                        action_probs,
                        opponent_influence,
                        self.nb_players,
                        self.nb_actions,
                        self.exploration_coefs)
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
    games = 10
    plays_per_game = 10
    opponent_class = pommerman.agents.RandomAgent
    # 2 or 4
    nb_players = 4
    nb_actions = 6
    mcts_iterations = 100
    use_progressive_widening = False
    depth = None
    heuristic_func = None
    # depth = 12
    # heuristic_func = heuristic_evaluator
    exploration_coefs = [math.sqrt(2)] * nb_players
    state_evaluator = RandomRolloutStateEvaluator(nb_players, nb_actions, depth=depth, heuristic_func=heuristic_func)
    smmcts = SMMCTS(nb_players=nb_players,
                    nb_actions=nb_actions,
                    exploration_coefs=[math.sqrt(2)] * nb_players,
                    state_evaluator=state_evaluator,
                    use_progressive_widening=use_progressive_widening)
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
            env = PommermanWrappedEnv(use_cython=use_cython, agents=agents, seed=seed)
            action_space = env.action_space
            state = env.reset()
            done = False
            while not done:
                actions = env.act(state)
                agent_action = smmcts.infer(env, iterations=mcts_iterations, progress_bar=progress_bar)
                actions.insert(0, agent_action)
                state, rewards, done = env.step(actions)
            win = int(rewards[0] == 1)
            tie = int(rewards.count(rewards[0]) == len(rewards))
            win_rate += win
            tie_rate += tie
    win_rate /= games * plays_per_game
    tie_rate /= games * plays_per_game
    lose_rate = 1 - win_rate - tie_rate
    print(f"win rate = {win_rate * 100}%, tie rate = {tie_rate * 100}%, lose rate = {lose_rate * 100}%")

