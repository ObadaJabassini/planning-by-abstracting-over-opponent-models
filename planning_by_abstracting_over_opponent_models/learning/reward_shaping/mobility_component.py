# a slightly modified version of https://github.com/haidertom/Pommerman/blob/master/demonstration.ipynb

from collections import OrderedDict

from planning_by_abstracting_over_opponent_models.learning.reward_shaping.reward_shaping_component import \
    RewardShapingComponent


class MobilityComponent(RewardShapingComponent):

    def __init__(self, mobility_reward=0.005, buffer_length=80):
        super().__init__()
        self.mobility_reward = mobility_reward
        self.buffer_length = buffer_length
        self.last_positions = OrderedDict()

    def shape(self, curr_state, curr_action):
        pos = tuple(curr_state['position'])
        reward = 0
        if len(self.last_positions) > 0 and pos not in self.last_positions:
            reward = self.mobility_reward
        self.last_positions[pos] = True
        if len(self.last_positions) > self.buffer_length:
            self.last_positions.popitem(last=False)
        return reward

    def reset(self):
        super().reset()
        self.last_positions = OrderedDict()

