import random
from typing import List

import numpy as np
import cpommerman
import pommerman

from planning_by_abstracting_over_opponent_models.pommerman_env.agents.pommerman_agent import PommermanAgent
from planning_by_abstracting_over_opponent_models.pommerman_env.pommerman_base_env import PommermanBaseEnv


class PommermanCythonEnv(PommermanBaseEnv):

    def __init__(self, agents: List[PommermanAgent], seed, training_agent=0):
        super().__init__(len(agents))
        self._seed = seed
        self.seed(self._seed)
        self.agents = agents
        self.env_render = pommerman.make('PommeFFACompetition-v0', agents)
        self.env = cpommerman.make()
        if 0 <= training_agent <= 3:
            self.env.set_training_agent(training_agent)
        self.action_space = self.env.action_space

    @staticmethod
    def seed(s):
        np.random.seed(s)
        random.seed(s)

    def get_observations(self):
        obs = self.env.get_observations()
        step_count = self.env.get_step_count()
        for ob in obs:
            ob['step_count'] = step_count
        return obs

    def get_done(self):
        return self.env.get_done()

    def get_rewards(self):
        return self.transform_rewards(self.env.get_rewards())

    def step(self, actions):
        actions = np.asarray(actions).astype(np.uint8)
        self.env.step(actions)
        return self.get_observations(), self.get_rewards(), self.get_done()

    def act(self, state):
        return [self.agents[i].act(state[i], self.action_space) for i in range(1, self.nb_players)]

    def reset(self):
        self.seed(self._seed)
        self.env.reset()
        for agent in self.agents:
            agent.reset_agent()
        return self.get_observations()

    def render(self, mode=None):
        if mode is None:
            mode = 'human'
        self.env_render._init_game_state = self.env.get_json_info()
        self.env_render.set_json_info()
        return self.env_render.render(mode)

    def get_game_state(self):
        return self.env.get_state()

    def set_game_state(self, game_state):
        self.env.set_state(game_state)
