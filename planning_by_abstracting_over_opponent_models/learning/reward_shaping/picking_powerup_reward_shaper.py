from pommerman.constants import Item

from planning_by_abstracting_over_opponent_models.learning.reward_shaping.reward_shaping_component import \
    RewardShapingComponent


class PickingPowerupRewardShaper(RewardShapingComponent):
    def __init__(self, pick_power_reward=0.1):
        self.pick_power_reward = pick_power_reward
        self.prev_state = None

    def update(self, curr_state, curr_action):
        self.prev_state = curr_state

    def shape(self, curr_state, curr_action):
        if self.prev_state is not None:
            potential_power = self.prev_state['board'][curr_state['position']]
            picked_power = potential_power in [Item.ExtraBomb.value, Item.IncrRange.value, Item.Kick.value]
            if picked_power:
                return self.pick_power_reward
        return 0

    def reset(self):
        self.prev_state = None
