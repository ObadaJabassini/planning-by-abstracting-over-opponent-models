# a slightly modified version of https://github.com/haidertom/Pommerman/blob/master/demonstration.ipynb

import numpy as np

from planning_by_abstracting_over_opponent_models.learning.reward_shaping.reward_shaping_component import \
    RewardShapingComponent


def is_between(a, b, c):
    crossproduct = (c.y - a.y) * (b.x - a.x) - (c.x - a.x) * (b.y - a.y)
    epsilon = 0.0001
    # compare versus epsilon for floating point values, or != 0 if using integers
    if abs(crossproduct) > epsilon:
        return False

    dotproduct = (c.x - a.x) * (b.x - a.x) + (c.y - a.y) * (b.y - a.y)
    if dotproduct < 0:
        return False

    squaredlengthba = (b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y)
    if dotproduct > squaredlengthba:
        return False

    return True


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def scale(self, s):
        self.x *= s
        self.y *= s
        return self


class AvoidingFlameComponent(RewardShapingComponent):

    def __init__(self, on_flame_reward=-0.01):
        super().__init__()
        self.on_flame_reward = on_flame_reward

    def update(self, curr_state, curr_action):
        pass

    def shape(self, curr_state, curr_action):
        s = 0
        bombs_pose = np.argwhere(curr_state['bomb_life'] != 0)
        for bp in bombs_pose:
            def rot_deg90cw(point):
                new_point = [0, 0]
                new_point[0] = point[1]
                new_point[1] = -point[0]
                return new_point

            # print(type(bp))
            factor = 1 / curr_state['bomb_life'][tuple(bp)]  # inverse of time left
            blast_strength = curr_state['bomb_blast_strength'][tuple(bp)]

            # blast directions
            blast_n = Point(0, 1).scale(blast_strength)
            blast_s = Point(0, -1).scale(blast_strength)
            blast_w = Point(-1, 0).scale(blast_strength)
            blast_e = Point(1, 0).scale(blast_strength)

            # agent on blast direction?
            bp_pose = rot_deg90cw(bp)
            my_pose = rot_deg90cw(curr_state['position'])
            my_pose = Point(my_pose[0] - bp_pose[0], my_pose[1] - bp_pose[1])  # my pose relative to the bomb!
            on_blast_direct = is_between(blast_n, blast_s, my_pose) or is_between(blast_w, blast_e, my_pose)
            if on_blast_direct:
                s += self.on_flame_reward * factor
        return s

    def reset(self):
        pass
