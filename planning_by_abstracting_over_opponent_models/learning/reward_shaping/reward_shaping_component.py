import abc


class RewardShapingComponent(abc.ABC):

    def __init__(self):
        self.prev_state = None

    @abc.abstractmethod
    def shape(self, curr_state, curr_action):
        raise NotImplementedError()

    def update(self, curr_state, curr_action):
        self.prev_state = curr_state

    def shape_and_update(self, curr_state, curr_action):
        reward = self.shape(curr_state, curr_action)
        self.update(curr_state, curr_action)
        return reward

    def reset(self):
        self.prev_state = None
