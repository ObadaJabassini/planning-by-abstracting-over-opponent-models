import numpy as np


class TreeNode:
    def __init__(self, state, parent, is_terminal, value_estimate, nb_players, action_space_size, exploration_coefs):
        self.state = state
        self.parent = parent
        self.is_terminal = is_terminal
        self.value_estimate = value_estimate
        self.exploration_coefs = exploration_coefs
        self.visit_count = 1
        self.children = dict()
        self.player_estimations = [self.init_estimations(action_space_size)] * nb_players

    def init_estimations(self, action_space_size):
        init = np.zeros((action_space_size, 2), np.float)
        return init

    def select_best_actions(self):
        return tuple([self.select_best_action_player(player) for player in range(len(self.player_estimations))])

    def select_best_action_player(self, player):
        uct = self.uct(player)
        return uct.argmax()

    def uct(self, player):
        estimations = self.player_estimations[player]
        exploration_coef = self.exploration_coefs[player]
        x, n = estimations[0], estimations[1]
        uct = x / n + exploration_coef * np.sqrt(np.log(self.visit_count) / n)
        return uct

    def update_actions_estimates(self, actions, value_estimate):
        for i in range(len(actions)):
            a = actions[i]
            self.player_estimations[i][a, 0] += value_estimate[i]
            self.player_estimations[i][a, 1] += 1


class SMMCTS:
    def __init__(self,
                 initial_state,
                 nb_players,
                 action_space_size,
                 transition_model,
                 reward_model,
                 exploration_coefs):
        self.nb_players = nb_players
        self.action_space_size = action_space_size
        self.transition_model = transition_model
        self.reward_model = reward_model
        self.exploration_coefs = exploration_coefs
        value_estimate = reward_model(initial_state)
        self.root = TreeNode(initial_state, None, value_estimate, nb_players, action_space_size, exploration_coefs)
        self.root.visit_count = 0

    def update(self, current_node: TreeNode):
        if current_node.is_terminal:
            return current_node.value_estimate

        actions = current_node.select_best_actions()
        new_state, is_terminal, ground_truth_reward = self.transition_model(current_node.state, actions)
        if actions not in current_node.children:
            value_estimate = self.reward_model(new_state) if not is_terminal else ground_truth_reward
            current_node.children[actions] = TreeNode(new_state,
                                                      current_node,
                                                      is_terminal,
                                                      value_estimate,
                                                      self.nb_players,
                                                      self.action_space_size,
                                                      self.exploration_coefs)
            return value_estimate

        child = current_node.children[actions]
        value_estimate = self.update(child)
        current_node.visit_count += 1
        current_node.value_estimate += value_estimate
        current_node.update_actions_estimates(actions, value_estimate)

    def select(self):
        best_actions = self.root.select_best_actions()
        return best_actions

    def simulate(self, iterations=100):
        for _ in range(iterations):
            self.update(self.root)
        best_actions = self.select()
        return best_actions
