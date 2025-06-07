#!/bin/bash

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

mpirun -np 1 python ../scripts/train_mpc.py  --train_name 'mpc_integrated' \
                                           --rnd_seed 1 \
                                           --max_iters 6000 \
                                           --save_interval 100 \
                                           --use_mpc True \
                                           --mpc_weight 0.1 \
                                           --visualize 