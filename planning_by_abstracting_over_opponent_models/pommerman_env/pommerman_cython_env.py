import numpy as np
import cpommerman
import pommerman

from planning_by_abstracting_over_opponent_models.pommerman_env.base_pommerman_env import BasePommermanEnv


class PommermanCythonEnv(BasePommermanEnv):

    def __init__(self, agents, seed):
        assert len(agents) == 4
        self.agents = agents
        self.env = cpommerman.make()
        self.env.set_training_agent(0)
        np.random.seed(seed)
        self.env_render = pommerman.make('PommeFFACompetition-v0', agents)
        self.action_space = self.env.action_space

    def get_observations(self):
        return self.env.get_observations()

    def get_done(self):
        return self.env.get_done()

    def get_rewards(self):
        return self.env.get_rewards()

    def step(self, actions):
        actions = np.asarray(actions).astype(np.uint8)
        self.env.step(actions)
        return self.get_observations(), self.get_rewards(), self.get_done()

    def act(self, state):
        return [self.agents[i].act(state[i], self.action_space) for i in range(1, len(self.agents))]

    def reset(self):
        self.env.reset()
        return self.get_observations()

    def render(self, mode=None):
        if mode is None:
            mode = 'human'
        state = self.env.get_json_info()
        self.env_render._init_game_state = state
        self.env_render.set_json_info()
        return self.env_render.render(mode)

    def get_game_state(self):
        return self.env.get_state()

    def set_game_state(self, game_state):
        self.env.set_state(game_state)

    def get_features(self):
        features = self.env.get_features()
        features = features[0]
        return features
