mpirun -np 1 python ../scripts/train.py  --train_name 'new_training' \
                                        --rnd_seed 1 \
                                        --max_iters 6000 \
                                        --save_interval 100 \
                                        #--restore_from 'previous_ckpts'