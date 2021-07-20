# a slightly modified version of https://github.com/haidertom/Pommerman/blob/master/demonstration.ipynb

from planning_by_abstracting_over_opponent_models.learning.reward_shaping.reward_shaping_component import \
    RewardShapingComponent


class EnemyKilledComponent(RewardShapingComponent):
    def __init__(self, enemy_killed_reward=0.5):
        super().__init__()
        self.enemy_killed_reward = enemy_killed_reward

    def shape(self, curr_state, curr_action):
        if self.prev_state is not None:
            l1 = len(self.prev_state["alive"])
            l2 = len(curr_state["alive"])
            return self.enemy_killed_reward if l1 > l2 else 0
        return 0
