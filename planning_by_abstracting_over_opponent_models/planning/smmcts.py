import math
from functools import partial
from time import sleep
from typing import List

import pommerman
import torch

from planning_by_abstracting_over_opponent_models.planning.random_rollout_evaluator import RandomRolloutEvaluator
from planning_by_abstracting_over_opponent_models.planning.state_evaluator import StateEvaluator
from planning_by_abstracting_over_opponent_models.planning.tree_node import TreeNode


class SMMCTS:
    def __init__(self,
                 nb_players,
                 nb_actions,
                 exploration_coefs,
                 state_evaluator: StateEvaluator):
        self.nb_players = nb_players
        self.nb_actions = nb_actions
        self.state_evaluator = state_evaluator
        self.exploration_coefs = torch.as_tensor(exploration_coefs).view(nb_players, 1).repeat(1, nb_actions)

    def search(self, env, current_node: TreeNode):
        if current_node.is_terminal:
            return current_node.value_estimate
        # select
        actions = self.select(current_node)
        state, rewards, is_terminal, _ = env.step(actions)
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
        return node.select_best_actions()

    def expand(self, env, state, rewards, is_terminal, actions, current_node):
        expected_value, action_probs, opponent_influence = self.state_evaluator.evaluate(env)
        value_estimate = expected_value if not is_terminal else torch.as_tensor(rewards)
        current_node.children[actions] = TreeNode(state=state,
                                                  parent=current_node,
                                                  is_terminal=is_terminal,
                                                  value_estimate=value_estimate,
                                                  action_prob_estimate=action_probs,
                                                  opponent_influence=opponent_influence,
                                                  nb_players=self.nb_players,
                                                  nb_actions=self.nb_actions,
                                                  exploration_coefs=self.exploration_coefs)
        return value_estimate

    def backpropagate(self, node, actions, value_estimate):
        node.update_actions_estimates(actions, value_estimate)

    def simulate(self, env, iterations=100):
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
        initial_state = env.get_json_info()
        for _ in range(iterations):
            self.search(env, root)
            env._init_game_state = initial_state
            env.reset()
        best_actions = root.select_best_actions()
        best_action = best_actions[0]
        return best_action


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


    nb_players = 2
    nb_actions = 6
    iterations = 100
    depth = None
    heuristic_func = None
    # depth = 12
    # heuristic_func = heuristic_evaluator
    wait_time = 4
    exploration_coefs = [math.sqrt(2)] * nb_players
    state_evaluator = RandomRolloutEvaluator(nb_players, nb_actions, depth=depth, heuristic_func=heuristic_func)
    smmcts = SMMCTS(nb_players=nb_players,
                    nb_actions=nb_actions,
                    exploration_coefs=[math.sqrt(2)] * nb_players,
                    state_evaluator=state_evaluator)
    agents: List[pommerman.agents.BaseAgent] = [pommerman.agents.RandomAgent() for _ in range(nb_players - 1)]
    agents.insert(0, DummyAgent())
    env = pommerman.make('PommeFFACompetition-v0', agents)
    env.set_training_agent(0)
    state = env.reset()
    done = False
    move_map = {
        0: "Stop",
        1: "Up",
        2: "Down",
        3: "Left",
        4: "Right",
        5: "Bomb"
    }
    while not done:
        opponent_action = env.act(state)
        agent_action = smmcts.simulate(env, iterations=iterations)
        actions = [agent_action, *opponent_action]
        state, rewards, done, _ = env.step(actions)
        env.render()
        moves = [move_map[action] for action in actions]
        print(moves)
        sleep(wait_time)
    print(rewards)
