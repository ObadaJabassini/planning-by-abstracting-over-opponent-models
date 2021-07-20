from random import randint
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import imageio
import torch.nn.functional as F
from PIL import Image
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from planning_by_abstracting_over_opponent_models.config import cpu
from planning_by_abstracting_over_opponent_models.learning.pommerman_env_utils import create_agent_model, \
    str_to_agent
from planning_by_abstracting_over_opponent_models.pommerman_env.agents.rl_agent import RLAgent
from planning_by_abstracting_over_opponent_models.pommerman_env.pommerman_cython_env import PommermanCythonEnv


class PassSecondLayer(nn.Module):
    def forward(self, inp):
        return inp[1]


if __name__ == '__main__':
    model_iteration = 25
    nb_opponents = 3
    device = cpu
    opponent_classes = ["simple", "simple", "simple"]
    combined_opponent_classes = ",".join(opponent_classes)
    opponent_classes = [str_to_agent(oc) for oc in opponent_classes]
    agent_model = create_agent_model(rank=1,
                                     seed=randint(1, 1000),
                                     nb_actions=6,
                                     nb_opponents=nb_opponents,
                                     nb_conv_layers=4,
                                     nb_filters=32,
                                     latent_dim=128,
                                     nb_soft_attention_heads=None,
                                     hard_attention_rnn_hidden_size=None,
                                     approximate_hard_attention=True,
                                     device=device,
                                     train=False)
    agent_model.load_state_dict(
        torch.load(f"../saved_models/{combined_opponent_classes}/agent_model_{model_iteration}.pt"))
    agent_model.eval()
    models = [nn.Sequential(agent_model.features_extractor, agent_model.opponent_models[i], PassSecondLayer()) for i in range(3)]
    cams = [GradCAM(model=models[i], target_layer=models[i][0].conv[-3], use_cuda=False) for i in range(3)]
    games = 1
    # for i in range(games):
    seed = randint(1, 1000)
    agents = [opponent_class() for opponent_class in opponent_classes]
    agent = RLAgent(0, agent_model, stochastic=True)
    agents.insert(0, agent)
    env = PommermanCythonEnv(agents, seed)
    action_space = env.action_space
    state = env.reset()
    rewards = [0] * (nb_opponents + 1)
    done = False
    step = 0
    # while not done:
    step += 1
    obs = env.get_features(state).to(device)
    obs = obs[0]
    obs = obs.unsqueeze(0)
    for i in range(3):
        cam = cams[i]
        result = cam(input_tensor=obs)
        result = result[0, :]
        plt.imshow(result)
        plt.savefig(f"cams/opponent_{i + 1}/{step}.png")
    # visualization = show_cam_on_image(r, grayscale_cam)
    # plt.imshow(visualization)
    # plt.show()
    agent_action = agent.act(obs, action_space)
    opponents_action = env.act(state)
    actions = [agent_action, *opponents_action]
    state, rewards, done = env.step(actions)
    env.render()
    # sleep(3)