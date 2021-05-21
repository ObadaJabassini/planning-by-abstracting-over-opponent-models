# heavily inspired by https://github.com/ikostrikov/pytorch-a3c/blob/master/main.py

import os
import argparse
import warnings
from multiprocessing import cpu_count

import pommerman.agents
import torch
import torch.multiprocessing as mp

from planning_by_abstracting_over_opponent_models.learning.config import cpu
from planning_by_abstracting_over_opponent_models.planning.modified_simple_agent import ModifiedSimpleAgent
from planning_by_abstracting_over_opponent_models.learning.pommerman_env_utils import create_agent_model
from planning_by_abstracting_over_opponent_models.learning.shared_adam import SharedAdam
from planning_by_abstracting_over_opponent_models.learning.train import train

warnings.filterwarnings('ignore')
torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=32)
parser.add_argument('--nb-processes', type=int, default=cpu_count() - 1, help='how many training processes to use')
parser.add_argument('--nb-episodes', type=int, default=int(2))
parser.add_argument('--nb-players', type=int, default=4, choices=[2, 4])
parser.add_argument('--nb-steps', type=int, default=20)
parser.add_argument('--use-simple-agent', dest="use_simple_agent", action="store_true")
parser.add_argument('--use-random-agent', dest="use_simple_agent", action="store_false")
parser.add_argument('--nb-conv-layers', type=int, default=3, choices=[3, 4])
parser.add_argument('--nb-filters', type=int, default=32)
parser.add_argument('--latent-dim', type=int, default=64)
parser.add_argument('--head-dim', type=int, default=64)
parser.add_argument('--nb-soft-attention-heads', type=int, default=None)
parser.add_argument('--hard-attention-rnn-hidden-size', type=int, default=None)
parser.add_argument('--shared-opt', dest='shared_opt', action='store_true')
parser.add_argument('--no-shared-opt', dest='shared_opt', action='store_false')
parser.set_defaults(use_simple_agent=True)
parser.set_defaults(shared_opt=True)


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    mp.set_start_method('spawn')
    args = parser.parse_args()
    device = cpu
    seed = args.seed
    use_cython = args.nb_players == 4
    nb_processes = args.nb_processes
    nb_episodes = args.nb_episodes
    nb_opponents = args.nb_players - 1
    opponent_class = ModifiedSimpleAgent if args.use_simple_agent else pommerman.agents.RandomAgent
    nb_steps = args.nb_steps
    model_spec = {
        "nb_conv_layers": args.nb_conv_layers,
        "nb_filters": args.nb_filters,
        "latent_dim": args.latent_dim,
        "head_dim": args.head_dim,
        "nb_soft_attention_heads": args.nb_soft_attention_heads,
        "hard_attention_rnn_hidden_size": args.hard_attention_rnn_hidden_size
    }
    nb_actions = 6
    shared_model = create_agent_model(nb_processes,
                                      seed,
                                      nb_actions,
                                      nb_opponents,
                                      device,
                                      **model_spec,
                                      train=True)
    shared_model.share_memory()
    optimizer = None
    if args.shared_opt:
        optimizer = SharedAdam(shared_model.parameters(),
                               lr=1e-4,
                               betas=(0.9, 0.999),
                               eps=1e-8,
                               weight_decay=1e-5)
        optimizer.share_memory()
    processes = []
    counter = mp.Value('i', 0)
    lock = mp.Lock()
    # args = (nb_processes + 1,
    #         seed,
    #         use_cython,
    #         shared_model,
    #         counter,
    #         model_spec,
    #         nb_episodes,
    #         nb_actions,
    #         nb_opponents,
    #         opponent_class,
    #         device)
    # p = mp.Process(target=test, args=args)
    # p.start()
    # processes.append(p)
    for rank in range(nb_processes):
        args = (rank,
                seed,
                use_cython,
                shared_model,
                counter,
                lock,
                model_spec,
                nb_episodes,
                nb_actions,
                nb_opponents,
                opponent_class,
                nb_steps,
                device,
                optimizer)
        p = mp.Process(target=train, args=args)
        p.start()
        processes.append(p)
    print("Started training")
    for p in processes:
        p.join()
    print("Saving the model..")
    torch.save(shared_model.state_dict(), "./agent_model.pt")
