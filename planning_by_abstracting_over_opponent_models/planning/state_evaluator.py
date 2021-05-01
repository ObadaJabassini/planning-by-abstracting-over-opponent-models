import abc


class StateEvaluator(abc.ABC):

    @abc.abstractmethod
    def evaluate(self, env, state):
        raise NotImplementedError()
