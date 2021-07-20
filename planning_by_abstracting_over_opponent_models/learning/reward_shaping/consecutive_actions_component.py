# a slightly modified version of https://github.com/haidertom/Pommerman/blob/master/demonstration.ipynb

from planning_by_abstracting_over_opponent_models.learning.reward_shaping.reward_shaping_component import \
    RewardShapingComponent


class ConsecutiveActionsComponent(RewardShapingComponent):

    def __init__(self, consecutive_actions_reward=-0.01):
        super().__init__()
        self.consecutive_actions_reward = consecutive_actions_reward
        self.prev_action = None
        self.cons_action_counter = 0

    def update(self, curr_state, curr_action):
        self.prev_action = curr_action

    def shape(self, curr_state, curr_action):
        if self.prev_action is not None:
            if curr_action == self.prev_action:
                self.cons_action_counter += 1
            else:
                self.cons_action_counter = 0
            if self.cons_action_counter >= 10:
                return self.consecutive_actions_reward
        return 0

    def reset(self):
        self.prev_action = None
        self.cons_action_counter = 0

