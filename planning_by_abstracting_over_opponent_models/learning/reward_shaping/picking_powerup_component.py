# a slightly modified version of https://github.com/haidertom/Pommerman/blob/master/demonstration.ipynb

from pommerman.constants import Item

from planning_by_abstracting_over_opponent_models.learning.reward_shaping.reward_shaping_component import \
    RewardShapingComponent


class PickingPowerupComponent(RewardShapingComponent):
    def __init__(self, pick_powerup_reward=0.02):
        super().__init__()
        self.pick_powerup_reward = pick_powerup_reward

    def shape(self, curr_state, curr_action):
        if self.prev_state is not None:
            potential_power = self.prev_state['board'][curr_state['position']]
            picked_power = potential_power in [Item.ExtraBomb.value, Item.IncrRange.value, Item.Kick.value]
            if picked_power:
                return self.pick_powerup_reward
        return 0
