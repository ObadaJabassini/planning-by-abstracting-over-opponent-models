import abc


class PommermanBaseEnv(abc.ABC):

    @abc.abstractmethod
    def get_observations(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_done(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_rewards(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def step(self, actions):
        raise NotImplementedError()

    @abc.abstractmethod
    def act(self, state):
        raise NotImplementedError()

    @abc.abstractmethod
    def render(self, mode=None):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_game_state(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def set_game_state(self, game_state):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_features(self):
        raise NotImplementedError()
