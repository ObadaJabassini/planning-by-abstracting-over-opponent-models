# a slightly modified version of https://github.com/haidertom/Pommerman/blob/master/demonstration.ipynb

import numpy as np
from pommerman.constants import Item


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


class RewardShaper:
    def __init__(self,
                 mobility_reward=0.1,
                 consecutive_actions_reward=-0.0001,
                 not_using_ammo_reward=-0.0001,
                 plant_bomb_near_wood_reward=0.05,
                 plant_bomb_near_enemy_reward=0.1,
                 avoid_bomb_reward=0.05,
                 on_flame_reward=-0.0001,
                 pick_power_reward=0.1,
                 catch_enemy_reward=0.001):
        self.mobility_reward = mobility_reward
        self.consecutive_actions_reward = consecutive_actions_reward
        self.not_using_ammo_reward = not_using_ammo_reward
        self.plant_bomb_near_wood_reward = plant_bomb_near_wood_reward
        self.plant_bomb_near_enemy_reward = plant_bomb_near_enemy_reward
        self.avoid_bomb_reward = avoid_bomb_reward
        self.on_flame_reward = on_flame_reward
        self.pick_power_reward = pick_power_reward
        self.catch_enemy_reward = catch_enemy_reward
        self.reset()

    def reset(self):
        self.prev_state = None
        self.prev_action = None
        self.cons_action_counter = 0
        self.not_using_ammo_counter = 0
        self.dist2bombs_prev = 0
        self.closest_enemy_id_prev = -1
        self.closest_enemy_dist_prev = float("inf")

    def shape(self, curr_state, curr_action, original_reward):
        if self.prev_state is None:
            self.prev_state = curr_state
            self.prev_action = curr_action
            return original_reward
        prev_state = self.prev_state
        reward = 0

        # reward stage 1: mobility
        pose_t = np.array(curr_state['position'])
        pose_tm1 = np.array(prev_state['position'])
        move_dist = np.linalg.norm(pose_t - pose_tm1)
        reward += self.mobility_reward if move_dist > 0 else 0

        # reward stage 2: consecutive actions
        if curr_action == self.prev_action:
            self.cons_action_counter += 1
        else:
            self.cons_action_counter = 0
        if self.cons_action_counter >= 10:
            reward += self.consecutive_actions_reward

        # reward stage 3: not using ammo
        if curr_state['ammo'] == prev_state['ammo']:
            self.not_using_ammo_counter += 1
        else:
            self.not_using_ammo_counter = 0
        if self.not_using_ammo_counter >= 10:
            reward += self.not_using_ammo_reward

        # stage 3: planting a bomb
        bombs_pose = np.argwhere(curr_state['bomb_life'] != 0)
        if curr_state['ammo'] < prev_state['ammo']:
            surroundings = [(-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0)]
            mybomb_pose = prev_state['position']  # equal to agent previous position
            # validate if the bomb actually exists there
            found_the_bomb = False
            for bp in bombs_pose:
                if np.equal(bp, mybomb_pose).all():
                    found_the_bomb = True
                    break
            assert found_the_bomb  # end of validation
            nr_woods = 0
            nr_enemies = 0
            for p in surroundings:
                cell_pose = (mybomb_pose[0] + p[0], mybomb_pose[1] + p[1])
                if cell_pose[0] > 10 or cell_pose[1] > 10:  # bigger than board size
                    continue
                nr_woods += curr_state['board'][cell_pose] == Item.Wood.value
                nr_enemies += curr_state['board'][cell_pose] in [e for e in curr_state['enemies']]
            assert nr_woods + nr_enemies < 10
            reward += self.plant_bomb_near_wood_reward * nr_woods + self.plant_bomb_near_enemy_reward * nr_enemies

        # stage 4: avoid flame
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
                reward += self.on_flame_reward * factor

        # stage 5: avoid bombs
        dist2bombs = 0
        for bp in bombs_pose:
            dist2bombs += np.linalg.norm(curr_state['position'] - bp)
        dist_delta = dist2bombs - self.dist2bombs_prev
        self.dist2bombs_prev = dist2bombs
        if dist_delta > 0 and move_dist:
            reward += dist_delta * self.avoid_bomb_reward

        # stage 6: pick a powerup
        potential_power = prev_state['board'][curr_state['position']]
        picked_power = (potential_power == Item.ExtraBomb.value) or \
                       (potential_power == Item.IncrRange.value) or \
                       (potential_power == Item.Kick.value)
        if picked_power:
            reward += self.pick_power_reward

        # stage 7: catch an enemy
        def closest_enemy():
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

        closest_enemy_id_cur, closest_enemy_dist_cur = closest_enemy()
        if self.closest_enemy_id_prev != closest_enemy_id_cur:
            self.closest_enemy_id_prev = closest_enemy_id_cur
            self.closest_enemy_dist_prev = closest_enemy_dist_cur
        else:
            catching_trhe = 4  # consider catching when close at most this much to the enemy
            if closest_enemy_dist_cur < self.closest_enemy_dist_prev and closest_enemy_dist_cur < catching_trhe:
                reward += self.catch_enemy_reward
                self.closest_enemy_dist_prev = closest_enemy_dist_cur
            if closest_enemy_dist_cur <= 1.1:  # got that close
                self.closest_enemy_dist_prev = float("inf")

        self.prev_state = curr_state
        self.prev_action = curr_action
        reward = np.clip(reward, -0.9, 0.9)
        return reward
