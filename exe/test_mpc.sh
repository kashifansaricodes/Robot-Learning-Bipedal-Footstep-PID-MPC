#!/bin/bash

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

python ../scripts/play_mpc.py --test_model 'versatile_walking' --use_mpc True 