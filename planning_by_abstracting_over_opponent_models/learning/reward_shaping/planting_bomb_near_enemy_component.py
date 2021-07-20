# a slightly modified version of https://github.com/haidertom/Pommerman/blob/master/demonstration.ipynb


import numpy as np

from planning_by_abstracting_over_opponent_models.learning.reward_shaping.reward_shaping_component import \
    RewardShapingComponent


class PlantingBombNearEnemyComponent(RewardShapingComponent):

    def __init__(self, plant_bomb_near_enemy_reward=0.05):
        super().__init__()
        self.plant_bomb_near_enemy_reward = plant_bomb_near_enemy_reward

    def shape(self, curr_state, curr_action):
        if self.prev_state is not None:
            bombs_pose = np.argwhere(curr_state['bomb_life'] != 0)
            if curr_state['ammo'] < self.prev_state['ammo']:
                # enemy_surroundings = [(-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0)]
                enemy_surroundings = [(0, 1), (1, 0), (0, -1), (-1, 0)]
                mybomb_pose = self.prev_state['position']  # equal to agent previous position
                # validate if the bomb actually exists there
                found_the_bomb = False
                for bp in bombs_pose:
                    if np.equal(bp, mybomb_pose).all():
                        found_the_bomb = True
                        break
                assert found_the_bomb  # end of validation
                nr_enemies = 0
                for p in enemy_surroundings:
                    cell_pose = (mybomb_pose[0] + p[0], mybomb_pose[1] + p[1])
                    if cell_pose[0] > 10 or cell_pose[1] > 10:  # bigger than board size
                        continue
                    nr_enemies += curr_state['board'][cell_pose] in [e for e in curr_state['enemies']]
                return self.plant_bomb_near_enemy_reward * nr_enemies
        return 0
