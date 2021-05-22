#!/bin/sh
nohup python -u learn.py --seed 32 --nb-episodes 1000000 --nb-steps 20 --save-interval 10000 --nb-players 4 --no-monitoring --use-simple-agent --shared-opt --use-cpu > results.txt 2>&1 &