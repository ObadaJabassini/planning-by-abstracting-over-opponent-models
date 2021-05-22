import abc


class StateEvaluator(abc.ABC):

    @abc.abstractmethod
    def evaluate(self, env):
        raise NotImplementedError()
