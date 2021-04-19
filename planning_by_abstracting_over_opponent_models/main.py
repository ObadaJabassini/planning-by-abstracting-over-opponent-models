import os
import torch
import torch.multiprocessing as mp

from planning_by_abstracting_over_opponent_models.config import cpu
from planning_by_abstracting_over_opponent_models.env import create_agent_model
from planning_by_abstracting_over_opponent_models.learning.agent_model import AgentModel
from planning_by_abstracting_over_opponent_models.learning.features_extractor import FeaturesExtractor
from planning_by_abstracting_over_opponent_models.learning.shared_adam import SharedAdam
from planning_by_abstracting_over_opponent_models.test import test
from planning_by_abstracting_over_opponent_models.train import train

torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    shared_optim = True
    device = cpu
    seed = 32
    nb_processes = 4
    action_space_size = 6
    max_steps = 800
    nb_opponents = 1
    rank = nb_processes
    shared_model = create_agent_model(seed,
                                      rank,
                                      action_space_size,
                                      nb_opponents,
                                      max_steps,
                                      device,
                                      return_agent=False)
    shared_model.share_memory()
    optimizer = None
    if shared_optim:
        optimizer = SharedAdam(shared_model.parameters(),
                               lr=1e-4,
                               betas=(0.9, 0.999),
                               eps=1e-8,
                               weight_decay=1e-5)
        optimizer.share_memory()

    processes = []
    counter = mp.Value('i', 0)
    lock = mp.Lock()
    args = (nb_processes, seed, shared_model, counter, action_space_size, nb_opponents, max_steps, device)
    p = mp.Process(target=test, args=args)
    p.start()
    processes.append(p)
    for rank in range(nb_processes):
        args = (rank, seed, shared_model, counter, lock, action_space_size, nb_opponents, max_steps, device, optimizer)
        p = mp.Process(target=train, args=args)
        p.start()
        processes.append(p)
    print("Started training")
    for p in processes:
        p.join()
