import gym


class TransitionModel:
    def __init__(self, env: gym.Env):
        self.env = env

    def __call__(self, state, actions):
        # should modify, this will not work
        self.env.state = state
        state, rewards, done, _ = self.env.step(actions)
        return state, not done, rewards
