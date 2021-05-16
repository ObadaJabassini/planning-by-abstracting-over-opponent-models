import numpy as np
import cpommerman
import pommerman

from planning_by_abstracting_over_opponent_models.pommerman_env.pommerman_base_env import PommermanBaseEnv


class PommermanCythonBaseEnv(PommermanBaseEnv):

    def __init__(self, agents, seed):
        super().__init__(len(agents))
        np.random.seed(seed)
        self.agents = agents
        self.env_render = pommerman.make('PommeFFACompetition-v0', agents)
        self.env = cpommerman.make()
        self.env.set_training_agent(0)
        self.action_space = self.env.action_space

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
        return [self.agents[i].act(state[i], self.action_space) for i in range(1, len(self.agents))]

    def reset(self):
        self.env.reset()
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
