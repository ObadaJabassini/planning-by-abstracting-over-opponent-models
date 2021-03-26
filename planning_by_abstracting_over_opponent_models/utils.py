import torch

gpu = torch.device("cuda:0")
cpu = torch.device("cpu")


def get_observation(state):
    board = state['board']
    board = torch.FloatTensor(board)
    board = board.unsqueeze(0).unsqueeze(0)
    board = board.to(gpu)
    return board
