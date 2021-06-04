import numpy as np

from planning_by_abstracting_over_opponent_models.learning.reward_shaping.reward_shaping_component import \
    RewardShapingComponent


class MobilityRewardShaper(RewardShapingComponent):

    def __init__(self, mobility_reward=0.1):
        self.mobility_reward = mobility_reward
        self.prev_state = None

    def update(self, curr_state, curr_action):
        self.prev_state = curr_state

    def shape(self, curr_state, curr_action):
        if self.prev_state is not None:
            pose_t = np.array(curr_state['position'])
            pose_tm1 = np.array(self.prev_state['position'])
            move_dist = np.linalg.norm(pose_t - pose_tm1)
            return self.mobility_reward if move_dist > 0 else 0
        return 0

    def reset(self):
        self.prev_state = None
