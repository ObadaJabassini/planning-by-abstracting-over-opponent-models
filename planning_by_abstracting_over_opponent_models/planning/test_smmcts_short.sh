#!/bin/sh
nohup python -u smmcts/smmcts.py --multiprocessing --nb-players 4 --nb-games 2 --nb-plays 5 --mcts-iterations 100 --fpu 1000 --search-opponent-actions --show-elapsed-time --no-progress-bar --use-simple-agent > results.txt 2>&1 &