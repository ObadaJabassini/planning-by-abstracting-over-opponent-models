#!/bin/sh
nohup python -u learn.py --seed 32 --nb-episodes 10000 --nb-steps 20 --save-interval 1000 --nb-players 4 --opponent-class static --no-monitoring --shared-opt --use-cpu > results.txt 2>&1 &