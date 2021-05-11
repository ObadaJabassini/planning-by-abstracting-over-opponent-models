#!/bin/sh
nohup python smmcts.py --multiprocessing --nb-players 4 --nb-games 10 --nb-plays 10 --mcts-iterations 1500 --use-cython --fpu 1000 --hide-elapsed-time --no-progress-bar --use-simple-agent &