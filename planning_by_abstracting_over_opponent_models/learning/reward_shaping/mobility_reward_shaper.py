from collections import OrderedDict

import numpy as np

from planning_by_abstracting_over_opponent_models.learning.reward_shaping.reward_shaping_component import \
    RewardShapingComponent


class MobilityRewardShaper(RewardShapingComponent):

    def __init__(self, mobility_reward=0.01):
        super().__init__()
        self.mobility_reward = mobility_reward
        self.last_positions = OrderedDict()

    def shape(self, curr_state, curr_action):
        pos = tuple(curr_state['position'])
        reward = 0
        if len(self.last_positions) > 0 and pos not in self.last_positions:
            reward = self.mobility_reward
        self.last_positions[pos] = True
        if len(self.last_positions) > 20:
            self.last_positions.popitem(last=False)
        return reward

    def reset(self):
        super().reset()
        self.last_positions = OrderedDict()

