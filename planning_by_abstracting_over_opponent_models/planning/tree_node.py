import math

import torch


class Player:
    def __init__(self, idd, nb_actions, action_probs_estimate, exploration_coef, fpu, pw_c=None, pw_alpha=None):
        self.idd = idd
        self.nb_actions = nb_actions
        self.action_estimations = torch.zeros(nb_actions)
        self.nb_action_visits = torch.zeros(nb_actions)
        self.exploration_coef = exploration_coef
        self.fpu = fpu
        self.pw_alpha = pw_alpha
        self.pw_c = pw_c
        self.use_progressive_widening = pw_alpha is not None
        if self.use_progressive_widening:
            self.action_probs_estimate, indices = torch.sort(action_probs_estimate)
            indices = indices.tolist()
            self.sorted_to_original_actions = {k: v for k, v in enumerate(indices)}
            self.original_to_sorted_actions = {v: k for k, v in self.sorted_to_original_actions.items()}
        else:
            self.action_probs_estimate = action_probs_estimate

    def most_visited_action(self):
        result = self.nb_action_visits.argmax().item()
        if self.use_progressive_widening:
            result = self.sorted_to_original_actions[result]
        return result

    def best_action(self, nb_visits):
        uct = self.compute_uct(nb_visits)
        best_action = uct.argmax().item()
        if self.use_progressive_widening:
            best_action = self.sorted_to_original_actions[best_action]
        return best_action

    def max_action(self):
        result = self.action_estimations.argmax().item()
        if self.use_progressive_widening:
            result = self.sorted_to_original_actions[result]
        return result

    def compute_uct(self, nb_visits):
        k = int(math.ceil(self.pw_c * (nb_visits ** self.pw_alpha))) if self.use_progressive_widening else self.nb_actions
        probs = self.action_probs_estimate[:k]
        c = self.exploration_coef
        x, n = self.action_estimations[:k], self.nb_action_visits[:k]
        x_bar = x / n
        exploration_term = c * probs * torch.sqrt(math.log2(nb_visits) / n)
        uct = x_bar + exploration_term
        # when an action is not explored, assign a fpu
        uct = torch.nan_to_num(uct, self.fpu, self.fpu, self.fpu)
        return uct

    def update_action_estimate(self, action, estimate):
        if self.use_progressive_widening:
            action = self.original_to_sorted_actions[action]
        self.action_estimations[action] += estimate
        self.nb_action_visits[action] += 1


class TreeNode:
    def __init__(self,
                 state,
                 parent,
                 is_terminal,
                 value_estimate,
                 action_prob_estimate,
                 nb_players,
                 nb_actions,
                 exploration_coefs,
                 fpus,
                 pw_cs=None,
                 pw_alphas=None):
        """
        :param state: the associated state
        :param parent: a pointer to the parent
        :param is_terminal: if the state is terminal
        :param value_estimate: the value estimate of the state, shape: (nb_players)
        :param action_prob_estimate: the probabilities of each action for each agent, shape: (nb_players, nb_actions)
        :param nb_players: the number of players
        :param nb_actions: the number of actions
        :param exploration_coefs: the exploration coefficient for each agent, shape: (nb_players, nb_actions)
        :param pw_alphas: progressive widening's alphas
        """
        self.state = state
        self.parent = parent
        self.is_terminal = is_terminal
        self.value_estimate = value_estimate
        self.nb_players = nb_players
        self.nb_actions = nb_actions
        self.exploration_coefs = exploration_coefs
        self.nb_visits = 1
        self.children = dict()
        if pw_alphas is None:
            pw_alphas = [None] * nb_players
        self.players = [Player(i,
                               nb_actions,
                               action_prob_estimate[i],
                               exploration_coefs[i],
                               fpus[i],
                               pw_cs[i],
                               pw_alphas[i]) for i in range(nb_players)]

    def most_visited_actions(self):
        return tuple((player.most_visited_action() for player in self.players))

    def best_actions(self):
        return tuple((player.best_action(self.nb_visits) for player in self.players))

    def update_actions_estimates(self, actions, action_value_estimate):
        """
        :param actions: the indices of the action that should be updated, shape: (nb_players)
        :param action_value_estimate: the estimated value function for each of those actions, shape: (nb_players)
        :return:
        """
        self.value_estimate += action_value_estimate
        self.nb_visits += 1
        for i in range(self.nb_players):
            self.players[i].update_action_estimate(actions[i], action_value_estimate[i])
