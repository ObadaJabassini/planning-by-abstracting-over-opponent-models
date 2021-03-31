import math

import torch


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
        """

        :param state: the associated state
        :param parent: a pointer to the parent
        :param is_terminal: if the state is terminal
        :param value_estimate: the value estimate of the state, shape: (nb_players)
        :param action_prob_estimate: the probabilities of each agent for each agent, shape: (nb_players, action_space_size)
        :param nb_players: the number of players
        :param action_space_size: the number of actions
        :param exploration_coefs: the exploration coefficient for each agent
        """
        self.state = state
        self.parent = parent
        self.is_terminal = is_terminal
        self.value_estimate = value_estimate
        self.action_prob_estimate = action_prob_estimate
        self.nb_players = nb_players
        self.exploration_coefs = exploration_coefs
        self.visit_count = 1
        self.children = dict()
        self.average_estimations = torch.zeros((nb_players, action_space_size))
        self.nb_action_visits = torch.zeros((nb_players, action_space_size))

    def select_best_actions(self):
        uct = self.uct()
        best_actions = torch.argmax(uct, dim=1)
        return tuple(best_actions.tolist())

    def uct(self):
        probs = self.action_prob_estimate
        c = self.exploration_coefs
        x, n = self.average_estimations, self.nb_action_visits
        x_bar = x / n
        exploration_term = torch.sqrt(math.log(self.visit_count) / n)
        # if there is no estimate for the action, assign zero
        x_bar = torch.nan_to_num(x_bar, 0, 0, 0)
        # when an action is not explored, assign a large value to ensure it will be explored
        exploration_term = torch.nan_to_num(exploration_term, 1000, 1000, 1000)
        exploration_term *= c * probs
        uct = x_bar + exploration_term
        return uct

    def update_actions_estimates(self, actions, action_value_estimate):
        """

        :param actions: the indicies of the action that should be updated, shape: (nb_players)
        :param action_value_estimate: the estimated value function for each of those actions, shape: (nb_players)
        :return:
        """
        r = range(len(actions))
        self.average_estimations[r, actions] += action_value_estimate
        self.nb_action_visits[r, actions] += 1
