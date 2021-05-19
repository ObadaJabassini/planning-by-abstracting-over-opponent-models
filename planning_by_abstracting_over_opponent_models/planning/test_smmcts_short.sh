#!/bin/sh
python smmcts.py --multiprocessing --nb-players 4 --nb-games 2 --nb-plays 5 --mcts-iterations 100 --fpu 1000 --hide-elapsed-time --no-progress-bar --use-simple-agent