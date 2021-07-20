# a slightly modified version of https://github.com/haidertom/Pommerman/blob/master/demonstration.ipynb

import numpy as np

from planning_by_abstracting_over_opponent_models.learning.reward_shaping.reward_shaping_component import \
    RewardShapingComponent


class CatchingEnemyComponent(RewardShapingComponent):

    def __init__(self, catch_enemy_reward=0.001):
        super().__init__()
        self.catch_enemy_reward = catch_enemy_reward
        self.closest_enemy_id_prev = -1
        self.closest_enemy_dist_prev = float("inf")

    @staticmethod
    def closest_enemy(curr_state):
        my_pose = curr_state['position']
        closest_enemy_id = -1
        closest_enemy_dist = float("inf")
        for e in curr_state['enemies']:
            enemy_pose = np.argwhere(curr_state['board'] == e)
            if len(enemy_pose) == 0:
                continue
            dist2_enemy = np.linalg.norm(my_pose - enemy_pose)
            if dist2_enemy <= closest_enemy_dist:
                closest_enemy_id = e
                closest_enemy_dist = dist2_enemy
        return closest_enemy_id, closest_enemy_dist

    def update(self, curr_state, curr_action):
        pass

    def shape(self, curr_state, curr_action):
        closest_enemy_id_cur, closest_enemy_dist_cur = self.closest_enemy(curr_state)
        reward = 0
        if self.closest_enemy_id_prev != closest_enemy_id_cur:
            self.closest_enemy_id_prev = closest_enemy_id_cur
            self.closest_enemy_dist_prev = closest_enemy_dist_cur
        else:
            catching_thre = 4  # consider catching when close at most this much to the enemy
            if closest_enemy_dist_cur < self.closest_enemy_dist_prev and closest_enemy_dist_cur < catching_thre:
                reward = self.catch_enemy_reward
                self.closest_enemy_dist_prev = closest_enemy_dist_cur
            if closest_enemy_dist_cur <= 1.1:  # got that close
                self.closest_enemy_dist_prev = float("inf")
        return reward

    def reset(self):
        self.closest_enemy_id_prev = -1
        self.closest_enemy_dist_prev = float("inf")
