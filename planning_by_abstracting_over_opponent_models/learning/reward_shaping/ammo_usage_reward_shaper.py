from planning_by_abstracting_over_opponent_models.learning.reward_shaping.reward_shaping_component import \
    RewardShapingComponent


class AmmoUsageRewardShaper(RewardShapingComponent):
    def __init__(self, not_using_ammo_reward=-0.0001):
        self.not_using_ammo_reward = not_using_ammo_reward
        self.prev_state = None
        self.not_using_ammo_counter = 0

    def update(self, curr_state, curr_action):
        self.prev_state = curr_state

    def shape(self, curr_state, curr_action):
        if self.prev_state is not None:
            if curr_state['ammo'] == self.prev_state['ammo']:
                self.not_using_ammo_counter += 1
            else:
                self.not_using_ammo_counter = 0
            if self.not_using_ammo_counter >= 10:
                return self.not_using_ammo_reward
        return 0

    def reset(self):
        self.prev_state = None
        self.not_using_ammo_counter = 0

