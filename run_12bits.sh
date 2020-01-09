#!/usr/bin/env bash
N=12
for (( i=0; i<N; i++ ))
do
    echo "begin predicting for bit $i"
    CUDA_VISIBLE_DEVICES=4,5,6,7 python -u run.py --epochs=5 --batch_size=16384 --predict=$(i) | tee raw_data_1bit_number_$(1).txt
done
