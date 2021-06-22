import abc


class PolicyEstimator(abc.ABC):

    @abc.abstractmethod
    def estimate(self, env):
        raise NotImplementedError()
