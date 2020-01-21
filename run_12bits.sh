#!/usr/bin/env bash
N=12
for (( i=0; i<N; i++ ))
do
    echo "begin predicting for bit $i"
    python -u run.py --epochs=2 --batch_size=32768 --predict=$i --seqlen=30| tee ./logs/rawdata_small_1bit_number_$i.txt
done
