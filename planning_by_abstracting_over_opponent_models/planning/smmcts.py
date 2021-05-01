import torch

from planning_by_abstracting_over_opponent_models.planning.state_evaluator import StateEvaluator
from planning_by_abstracting_over_opponent_models.planning.tree_node import TreeNode


class SMMCTS:
    def __init__(self,
                 nb_players,
                 action_space_size,
                 exploration_coefs,
                 state_evaluator: StateEvaluator):
        self.nb_players = nb_players
        self.action_space_size = action_space_size
        self.state_evaluator = state_evaluator
        self.exploration_coefs = torch.as_tensor(exploration_coefs).view(nb_players, 1)

    def update(self, env, current_node: TreeNode):
        if current_node.is_terminal:
            return current_node.value_estimate
        # select
        actions = current_node.select_best_actions()
        state, rewards, is_terminal, _ = env.step(actions)
        # expand
        if actions not in current_node.children:
            expected_value, action_probs, opponent_influence = self.state_evaluator.evaluate(env)
            value_estimate = expected_value if not is_terminal else rewards
            current_node.children[actions] = TreeNode(state,
                                                      current_node,
                                                      is_terminal,
                                                      value_estimate,
                                                      action_probs,
                                                      opponent_influence,
                                                      self.nb_players,
                                                      self.action_space_size,
                                                      self.exploration_coefs)
        else:
            child = current_node.children[actions]
            value_estimate = self.update(env, child)
        # backpropagate
        current_node.update_actions_estimates(actions, value_estimate)
        return value_estimate

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
                        self.action_space_size,
                        self.exploration_coefs)
        env._init_game_state = env.get_json_info()
        for _ in range(iterations):
            env.reset()
            self.update(env, root)
        env.reset()
        env._init_game_state = None
        best_actions = root.select_best_actions()
        best_action = best_actions[0]
        return best_action
