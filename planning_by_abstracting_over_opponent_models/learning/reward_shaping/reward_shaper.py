# a slightly modified version of https://github.com/haidertom/Pommerman/blob/master/demonstration.ipynb
from typing import List

import numpy as np

from planning_by_abstracting_over_opponent_models.learning.reward_shaping.reward_shaping_component import \
    RewardShapingComponent


class RewardShaper:
    def __init__(self, reward_shaping_components: List[RewardShapingComponent]):
        self.reward_shaping_components = reward_shaping_components

    def reset(self):
        for comp in self.reward_shaping_components:
            comp.reset()

    def shape(self, curr_state, curr_action):
        reward = 0
        for comp in self.reward_shaping_components:
            reward += comp.shape_and_update(curr_state, curr_action)
        reward = np.clip(reward, -0.9, 0.9)
        return reward
