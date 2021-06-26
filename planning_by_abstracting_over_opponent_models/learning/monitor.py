import time

import torch


def monitor(shared_model,
            opponent_classes,
            save_interval):
    combined_opponent_classes = ",".join(opponent_classes)
    counter = 1
    while True:
        time.sleep(save_interval)
        torch.save(shared_model.state_dict(), f"../saved_models/{combined_opponent_classes}/agent_model_{counter}.pt")
        counter += 1
