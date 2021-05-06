import math

import torch
from pommerman.envs.v0 import Pomme


class TreeNode:
    def __init__(self,
                 state,
                 parent,
                 is_terminal,
                 value_estimate,
                 action_prob_estimate,
                 opponent_influence,
                 nb_players,
                 nb_actions,
                 exploration_coefs,
                 use_progressive_widening=False):
        """
        :param state: the associated state
        :param parent: a pointer to the parent
        :param is_terminal: if the state is terminal
        :param value_estimate: the value estimate of the state, shape: (nb_players)
        :param action_prob_estimate: the probabilities of each action for each agent, shape: (nb_players, nb_actions)
        :param nb_players: the number of players
        :param nb_actions: the number of actions
        :param exploration_coefs: the exploration coefficient for each agent, shape: (nb_players, nb_actions)
        """
        self.state = state
        self.parent = parent
        self.is_terminal = is_terminal
        self.value_estimate = value_estimate
        self.action_prob_estimate = action_prob_estimate
        self.opponent_influence = opponent_influence
        self.nb_players = nb_players
        self.nb_actions = nb_actions
        self.exploration_coefs = exploration_coefs
        self.use_progressive_widening = use_progressive_widening
        self.nb_visits = 1
        self.children = dict()
        self.average_estimations = torch.zeros(nb_players, nb_actions)
        self.nb_action_visits = torch.zeros(nb_players, nb_actions)

    def most_visited_actions(self):
        most_visited_actions = self.nb_action_visits.argmax(dim=1).tolist()
        return tuple(most_visited_actions)

    def best_actions(self):
        if self.use_progressive_widening:
            ks = torch.pow(self.nb_visits, self.opponent_influence).ceil().type(torch.LongTensor).tolist()
            # consider all agent's actions
            ks.insert(0, self.nb_actions)
        else:
            # consider all players' actions
            ks = [self.nb_actions] * self.nb_players
        best_actions = []
        for player, k in enumerate(ks):
            uct = self.compute_uct_for_player(player, k)
            best_action = uct.argmax().item()
            best_actions.append(best_action)
        return tuple(best_actions)

    def compute_uct_for_player(self, player, k):
        probs = self.action_prob_estimate[player, :k]
        c = self.exploration_coefs[player]
        x, n = self.average_estimations[player, :k], self.nb_action_visits[player, :k]
        x_bar = x / n
        exploration_term = torch.sqrt(math.log2(self.nb_visits) / n)
        # if there is no estimate for the action, assign zero
        x_bar = torch.nan_to_num(x_bar, 0, 0, 0)
        exploration_term *= c * probs
        # when an action is not explored, assign a large value to ensure it will be explored
        exploration_term = torch.nan_to_num(exploration_term, 1000, 1000, 1000)
        uct = x_bar + exploration_term
        return uct

    def update_actions_estimates(self, actions, action_value_estimate):
        """
        :param actions: the indicies of the action that should be updated, shape: (nb_players)
        :param action_value_estimate: the estimated value function for each of those actions, shape: (nb_players)
        :return:
        """
        self.value_estimate += action_value_estimate
        self.nb_visits += 1
        r = range(self.nb_players)
        self.average_estimations[r, actions] += action_value_estimate
        self.nb_action_visits[r, actions] += 1
