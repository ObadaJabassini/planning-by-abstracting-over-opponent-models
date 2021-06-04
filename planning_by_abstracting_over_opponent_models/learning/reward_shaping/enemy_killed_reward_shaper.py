from planning_by_abstracting_over_opponent_models.learning.reward_shaping.reward_shaping_component import \
    RewardShapingComponent


class EnemyKilledRewardShaper(RewardShapingComponent):
    def __init__(self, enemy_killed_reward=0.5):
        self.enemy_killed_reward = enemy_killed_reward
        self.prev_state = None

    def update(self, curr_state, curr_action):
        self.prev_state = curr_state

    def shape(self, curr_state, curr_action):
        if self.prev_state is not None:
            l1 = len(self.prev_state["alive"])
            l2 = len(curr_state["alive"])
            return self.enemy_killed_reward if l2 < l1 else 0
        return 0

    def reset(self):
        self.prev_state = None
