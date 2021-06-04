import abc


class RewardShapingComponent(abc.ABC):
    @abc.abstractmethod
    def shape(self, curr_state, curr_action):
        raise NotImplementedError()

    @abc.abstractmethod
    def update(self, curr_state, curr_action):
        raise NotImplementedError()

    def shape_and_update(self, curr_state, curr_action):
        reward = self.shape(curr_state, curr_action)
        self.update(curr_state, curr_action)
        return reward

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError()
