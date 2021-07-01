from planning_by_abstracting_over_opponent_models.planning.player import MCTSPlayer, RandomPlayer


class TreeNode:
    def __init__(self,
                 state,
                 parent,
                 is_terminal,
                 value_estimate,
                 action_probs_estimate,
                 nb_players,
                 nb_actions,
                 exploration_coefs,
                 fpus,
                 random_players,
                 pw_cs,
                 pw_alphas):
        """
        :param state: the associated state
        :param parent: a pointer to the parent
        :param is_terminal: if the state is terminal
        :param value_estimate: the value estimate of the state, shape: (nb_players)
        :param action_probs_estimate: the probabilities of each action for each agent, shape: (nb_players, nb_actions)
        :param nb_players: the number of players
        :param nb_actions: the number of actions
        :param exploration_coefs: the exploration coefficient for each agent, shape: (nb_players, nb_actions)
        :param pw_alphas: progressive widening alphas
        """
        self.state = state
        self.parent = parent
        self.is_terminal = is_terminal
        self.value_estimate = value_estimate
        self.nb_players = nb_players
        self.nb_visits = 1
        self.children = dict()
        self.players = []
        for i in range(nb_players):
            player = MCTSPlayer(i,
                                nb_actions,
                                action_probs_estimate[i],
                                exploration_coefs[i],
                                fpus[i],
                                pw_cs[i],
                                pw_alphas[i]) if not random_players[i] else RandomPlayer(i, nb_actions)
            self.players.append(player)

    def most_visited_actions(self):
        return tuple((player.most_visited_action() for player in self.players))

    def best_actions(self):
        return tuple((player.best_action(self.nb_visits) for player in self.players))

    def max_actions(self):
        return tuple((player.max_action() for player in self.players))

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
