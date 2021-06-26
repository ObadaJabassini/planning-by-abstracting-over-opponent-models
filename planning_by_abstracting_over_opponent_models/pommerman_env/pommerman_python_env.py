import pommerman

from planning_by_abstracting_over_opponent_models.pommerman_env.pommerman_base_env import PommermanBaseEnv


class PommermanPythonEnv(PommermanBaseEnv):

    def __init__(self, agents, seed):
        super().__init__(len(agents))
        self.nb_players = len(agents)
        self.env = pommerman.make('PommeFFACompetition-v0', agents)
        self.env.seed(seed)
        self.env.set_training_agent(0)
        self.action_space = self.env.action_space

    def get_observations(self):
        obs = self.env.get_observations()
        return obs

    def get_done(self):
        return self.env._get_done()

    def get_rewards(self):
        rewards = self.env._get_rewards()
        rewards = self.transform_rewards(rewards)
        return rewards

    def reset(self):
        self.env.reset()
        return self.get_observations()

    def step(self, actions):
        state, rewards, done, _ = self.env.step(actions)
        rewards = self.transform_rewards(rewards)
        return state, rewards, done

    def act(self, state):
        return self.env.act(state)

    def render(self, mode=None):
        if mode is None:
            mode = 'human'
        return self.env.render(mode)

    def get_game_state(self):
        return self.env.get_json_info()

    def set_game_state(self, game_state):
        self.env._init_game_state = game_state
        self.env.set_json_info()
