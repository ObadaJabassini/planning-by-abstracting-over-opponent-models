# a slightly modified version of https://github.com/haidertom/Pommerman/blob/master/demonstration.ipynb

from pommerman.constants import Action

from planning_by_abstracting_over_opponent_models.learning.reward_shaping.reward_shaping_component import \
    RewardShapingComponent


class AvoidingIllegalMoves(RewardShapingComponent):

    def __init__(self, illegal_move_reward=-0.03):
        super().__init__()
        self.illegal_move_reward = illegal_move_reward

    def shape(self, curr_state, curr_action):
        if self.prev_state is not None:
            ammo = self.prev_state['ammo']
            if curr_action != Action.Bomb.value:
                if curr_state['position'] == self.prev_state['position']:
                    return self.illegal_move_reward
            elif ammo == 0:
                return self.illegal_move_reward
        return 0
