from functools import partial
import abc
import math
import random
import torch


class Player(abc.ABC):
    def __init__(self, idd):
        self.idd = idd

    @abc.abstractmethod
    def most_visited_action(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def best_action(self, nb_visits):
        raise NotImplementedError()

    @abc.abstractmethod
    def max_action(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def update_action_estimate(self, action, estimate):
        raise NotImplementedError()


class RandomPlayer(Player):

    def __init__(self, idd, nb_actions):
        super().__init__(idd)
        self.nb_actions = nb_actions
        self.rand = partial(random.randint, a=0, b=nb_actions - 1)

    def most_visited_action(self):
        return self.rand()

    def best_action(self, nb_visits):
        return self.rand()

    def max_action(self):
        return self.rand()

    def update_action_estimate(self, action, estimate):
        pass


class MCTSPlayer(Player):
    def __init__(self, idd, nb_actions, action_probs_estimate, exploration_coef, fpu, pw_c=None, pw_alpha=None):
        super().__init__(idd)
        self.nb_actions = nb_actions
        self.action_estimations = torch.zeros(nb_actions)
        self.nb_action_visits = torch.zeros(nb_actions)
        self.exploration_coef = exploration_coef
        self.fpu = fpu
        self.action_probs_estimate = action_probs_estimate
        self.use_progressive_widening = pw_alpha is not None and pw_c is not None
        if self.use_progressive_widening:
            self.pw_alpha = pw_alpha
            self.pw_c = pw_c
            self.action_probs_estimate, indices = torch.sort(self.action_probs_estimate, dim=-1, descending=True)
            indices = indices.tolist()
            self.sorted_to_original_actions = {k: v for k, v in enumerate(indices)}
            self.original_to_sorted_actions = {v: k for k, v in self.sorted_to_original_actions.items()}

    def _compute_k(self, nb_visits):
        k = self.pw_c * (nb_visits ** self.pw_alpha)
        k = math.ceil(k)
        k = max(k, 1)
        return k

    def best_action(self, nb_visits):
        k = self._compute_k(nb_visits) if self.use_progressive_widening else self.nb_actions
        k = int(k)
        result = self.uct(nb_visits, k)
        result = result.argmax().item()
        if self.use_progressive_widening:
            result = self.sorted_to_original_actions[result]
        return result

    def max_action(self):
        result = self.action_estimations.argmax().item()
        if self.use_progressive_widening:
            result = self.sorted_to_original_actions[result]
        return result

    def most_visited_action(self):
        result = self.nb_action_visits.argmax().item()
        if self.use_progressive_widening:
            result = self.sorted_to_original_actions[result]
        return result

    def uct(self, nb_visits, k):
        probs = self.action_probs_estimate[:k]
        x, n = self.action_estimations[:k], self.nb_action_visits[:k]
        exploitation_term = x / n
        exploration_term = self.exploration_coef * probs * torch.sqrt(math.log2(nb_visits) / n)
        uct = exploitation_term + exploration_term
        # when an action is not explored, assign fpu
        uct = torch.nan_to_num(uct, self.fpu, self.fpu, self.fpu)
        return uct

    def update_action_estimate(self, action, estimate):
        if self.use_progressive_widening:
            action = self.original_to_sorted_actions[action]
        self.action_estimations[action] += estimate
        self.nb_action_visits[action] += 1
