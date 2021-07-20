# a slightly modified version of https://github.com/haidertom/Pommerman/blob/master/demonstration.ipynb

import numpy as np

from planning_by_abstracting_over_opponent_models.learning.reward_shaping.reward_shaping_component import \
    RewardShapingComponent


class AvoidingBombComponent(RewardShapingComponent):

    def __init__(self, avoid_bomb_reward=0.05):
        super().__init__()
        self.avoid_bomb_reward = avoid_bomb_reward
        self.dist2bombs_prev = 0

    def shape(self, curr_state, curr_action):
        if self.prev_state is not None:
            dist2bombs = 0
            bombs_pose = np.argwhere(curr_state['bomb_life'] != 0)
            for bp in bombs_pose:
                dist2bombs += np.linalg.norm(curr_state['position'] - bp)
            dist_delta = dist2bombs - self.dist2bombs_prev
            self.dist2bombs_prev = dist2bombs
            pose_t = np.array(curr_state['position'])
            pose_tm1 = np.array(self.prev_state['position'])
            move_dist = np.linalg.norm(pose_t - pose_tm1)
            if dist_delta > 0 and move_dist > 0:
                return dist_delta * self.avoid_bomb_reward
        return 0

    def reset(self):
        super().reset()
        self.dist2bombs_prev = 0
