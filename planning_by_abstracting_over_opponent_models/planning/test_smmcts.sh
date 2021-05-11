#!/bin/sh
nohup python smmcts.py --multiprocessing --nb-players 2 --nb-games 10 --nb-plays 10 --mcts-iterations 200 --use-python --fpu 100 &