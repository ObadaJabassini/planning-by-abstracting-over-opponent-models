#!/bin/sh
nohup python -u smmcts.py --multiprocessing --nb-players 4 --nb-games 10 --nb-plays 10 --mcts-iterations 7500 --fpu 1000 --show-elapsed-time --no-progress-bar --use-simple-agent > results.txt 2>&1 &