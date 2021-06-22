import abc


class ValueEstimator(abc.ABC):

    @abc.abstractmethod
    def estimate(self, env):
        raise NotImplementedError()
