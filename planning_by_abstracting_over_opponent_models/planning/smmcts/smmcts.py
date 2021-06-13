import torch

from planning_by_abstracting_over_opponent_models.planning.smmcts.tree_node import TreeNode
from planning_by_abstracting_over_opponent_models.planning.smmcts.state_evaluator import StateEvaluator

torch.autograd.set_detect_anomaly(True)


class SMMCTS:
    def __init__(self,
                 nb_players,
                 nb_actions,
                 exploration_coefs,
                 state_evaluator: StateEvaluator):
        self.nb_players = nb_players
        self.nb_actions = nb_actions
        self.exploration_coefs = exploration_coefs
        self.state_evaluator = state_evaluator

    def search(self, env, current_node: TreeNode, fpus, random_players):
        if current_node.is_terminal:
            return current_node.value_estimate
        # select
        actions = self.select(current_node)
        state, rewards, is_terminal = env.step(actions)
        # expand
        if actions not in current_node.children:
            value_estimate = self.expand(env, state, rewards, is_terminal, actions, current_node, fpus, random_players)
        else:
            child = current_node.children[actions]
            value_estimate = self.search(env, child, fpus, random_players)
        # backpropagate
        self.backpropagate(current_node, actions, value_estimate)
        return value_estimate

    def select(self, node):
        return node.best_actions()

    def expand(self, env, state, rewards, is_terminal, actions, current_node, fpus, random_players):
        if is_terminal:
            value_estimate = torch.as_tensor(rewards)
            action_prob_estimate = torch.full((self.nb_players, self.nb_actions), 1 / self.nb_actions)
            pw_alphas = [None] * self.nb_players
            pw_cs = [None] * self.nb_players
        else:
            value_estimate, action_prob_estimate, pw_cs, pw_alphas = self.state_evaluator.evaluate(env)
        current_node.children[actions] = TreeNode(state=state,
                                                  parent=current_node,
                                                  is_terminal=is_terminal,
                                                  value_estimate=value_estimate,
                                                  action_probs_estimate=action_prob_estimate,
                                                  nb_players=self.nb_players,
                                                  nb_actions=self.nb_actions,
                                                  exploration_coefs=self.exploration_coefs,
                                                  fpus=fpus,
                                                  random_players=random_players,
                                                  pw_cs=pw_cs,
                                                  pw_alphas=pw_alphas)
        return value_estimate

    def backpropagate(self, node, actions, value_estimate):
        node.update_actions_estimates(actions, value_estimate)

    def infer(self, env, iterations, fpus, random_players):
        initial_state = env.get_observations()
        value_estimate, action_probs_estimate, pw_cs, pw_alphas = self.state_evaluator.evaluate(env)
        root = TreeNode(state=initial_state,
                        parent=None,
                        is_terminal=False,
                        value_estimate=value_estimate,
                        action_probs_estimate=action_probs_estimate,
                        nb_players=self.nb_players,
                        nb_actions=self.nb_actions,
                        exploration_coefs=self.exploration_coefs,
                        fpus=fpus,
                        random_players=random_players,
                        pw_cs=pw_cs,
                        pw_alphas=pw_alphas)
        game_state = env.get_game_state()
        for iteration in range(iterations):
            self.search(env, root, fpus, random_players)
            env.set_game_state(game_state)
        most_visited_action = root.most_visited_actions()[0]
        return most_visited_action
