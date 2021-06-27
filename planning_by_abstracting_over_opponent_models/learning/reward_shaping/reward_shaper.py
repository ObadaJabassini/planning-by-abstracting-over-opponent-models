# a slightly modified version of https://github.com/haidertom/Pommerman/blob/master/demonstration.ipynb
from typing import List

import numpy as np

from planning_by_abstracting_over_opponent_models.learning.reward_shaping.ammo_usage_component import AmmoUsageComponent
from planning_by_abstracting_over_opponent_models.learning.reward_shaping.avoiding_bomb_component import \
    AvoidingBombComponent
from planning_by_abstracting_over_opponent_models.learning.reward_shaping.avoiding_flame_component import \
    AvoidingFlameComponent
from planning_by_abstracting_over_opponent_models.learning.reward_shaping.avoiding_illegal_moves import \
    AvoidingIllegalMoves
from planning_by_abstracting_over_opponent_models.learning.reward_shaping.catching_enemy_component import \
    CatchingEnemyComponent
from planning_by_abstracting_over_opponent_models.learning.reward_shaping.consecutive_actions_component import \
    ConsecutiveActionsComponent
from planning_by_abstracting_over_opponent_models.learning.reward_shaping.enemy_killed_component import \
    EnemyKilledComponent
from planning_by_abstracting_over_opponent_models.learning.reward_shaping.mobility_component import MobilityComponent
from planning_by_abstracting_over_opponent_models.learning.reward_shaping.picking_powerup_component import \
    PickingPowerupComponent
from planning_by_abstracting_over_opponent_models.learning.reward_shaping.planting_bomb_near_enemy_component import \
    PlantingBombNearEnemyComponent
from planning_by_abstracting_over_opponent_models.learning.reward_shaping.planting_bomb_near_wall_component import \
    PlantingBombNearWallComponent
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


def strs_to_reward_shaper(strs):
    d = {
        "ammo_usage": AmmoUsageComponent,
        "avoiding_bomb": AvoidingBombComponent,
        "avoiding_flame": AvoidingFlameComponent,
        "catching_enemy": CatchingEnemyComponent,
        "consecutive_actions": ConsecutiveActionsComponent,
        "enemy_killed": EnemyKilledComponent,
        "mobility": MobilityComponent,
        "picking_powerup": PickingPowerupComponent,
        "planting_bomb_near_wall": PlantingBombNearWallComponent,
        "planting_bomb_near_enemy": PlantingBombNearEnemyComponent,
        "avoiding_illegal_moves": AvoidingIllegalMoves
    }
    return RewardShaper([d[s]() for s in strs])
