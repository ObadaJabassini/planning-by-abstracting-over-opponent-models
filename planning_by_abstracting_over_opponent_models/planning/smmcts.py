import math
from copy import deepcopy

import torch
import torch.nn.functional as F

from planning_by_abstracting_over_opponent_models.config import cpu


class TreeNode:
    def __init__(self,
                 state,
                 parent,
                 is_terminal,
                 value_estimate,
                 action_prob_estimate,
                 nb_players,
                 action_space_size,
                 exploration_coefs):
        self.state = state
        self.parent = parent
        self.is_terminal = is_terminal
        self.value_estimate = value_estimate
        self.action_prob_estimate = action_prob_estimate
        self.nb_players = nb_players
        self.exploration_coefs = exploration_coefs
        self.visit_count = 0
        self.children = dict()
        self.average_estimations = torch.zeros((nb_players, action_space_size))
        self.nb_action_visits = torch.zeros((nb_players, action_space_size))

    def select_best_actions(self):
        return tuple(self.select_best_action_player(player) for player in range(self.nb_players))

    def select_best_action_player(self, player):
        uct = self.uct(player)
        return uct.argmax()

    def uct(self, player):
        prob = self.action_prob_estimate[player]
        exploration_coef = self.exploration_coefs[player]
        x, n = self.average_estimations[player], self.nb_action_visits[player]
        uct = x / n + exploration_coef * prob * torch.sqrt(math.log(self.visit_count) / n)
        return uct

    def update_actions_estimates(self, actions, action_value_estimate):
        for i in range(len(actions)):
            a = actions[i]
            self.average_estimations[i][a] += action_value_estimate[i]
            self.nb_action_visits[i][a] += 1


class SMMCTS:
    def __init__(self,
                 nb_players,
                 action_space_size,
                 extract_observation_func,
                 agent_model,
                 exploration_coefs):
        self.nb_players = nb_players
        self.action_space_size = action_space_size
        self.extract_observation_func = extract_observation_func
        self.agent_model = agent_model
        self.exploration_coefs = exploration_coefs

    def update(self, env, current_node: TreeNode):
        if current_node.is_terminal:
            return current_node.value_estimate
        actions = current_node.select_best_actions()
        current_node.visit_count += 1
        state, rewards, is_terminal, _ = env.step(actions)

        if actions not in current_node.children:
            expected_value, action_probs = self.estimate_node(state)
            value_estimate = expected_value if not is_terminal else rewards
            current_node.children[actions] = TreeNode(state,
                                                      current_node,
                                                      is_terminal,
                                                      value_estimate,
                                                      action_probs,
                                                      self.nb_players,
                                                      self.action_space_size,
                                                      self.exploration_coefs)
            return value_estimate

        child = current_node.children[actions]
        value_estimate = self.update(env, child)
        # current_node.value_estimate += value_estimate
        current_node.update_actions_estimates(actions, value_estimate)
        return value_estimate

    def estimate_node(self, state):
        obs = self.extract_observation_func(state)
        agent_action_log, agent_value, opponents_action_log, opponent_values, opponent_influence = self.agent_model(obs)
        value_estimate = self.estimate_values(agent_value, opponent_values)
        action_probs_estimate = self.estimate_probabilities(agent_action_log, opponents_action_log)
        return value_estimate, action_probs_estimate

    def estimate_values(self, agent_value, opponent_values):
        agent_value = agent_value.view(-1).to(cpu)
        opponent_values = opponent_values.view(-1).to(cpu)
        value_estimate = torch.cat((agent_value, opponent_values))
        return value_estimate

    def estimate_probabilities(self, agent_action_log, opponent_action_log):
        agent_action_probs = F.softmax(agent_action_log, dim=-1)
        agent_action_probs = agent_action_probs.view((1, -1)).to(cpu)
        opponent_action_probs = F.softmax(opponent_action_log, dim=-1)
        opponent_action_probs = opponent_action_probs.view(self.nb_opponents, -1).to(cpu)
        probs = torch.vstack((agent_action_probs, opponent_action_probs))
        return probs

    def simulate(self, env, initial_state, iterations=100):
        value_estimate, action_probs = self.estimate_node(initial_state)
        root = TreeNode(initial_state,
                        None,
                        False,
                        value_estimate,
                        action_probs,
                        self.nb_players,
                        self.action_space_size,
                        self.exploration_coefs)
        for _ in range(iterations):
            snapshot = deepcopy(env)
            self.update(snapshot, root)
            env = snapshot
        best_actions = root.select_best_actions()
        best_action = best_actions[0]
        return best_action
