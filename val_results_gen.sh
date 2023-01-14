#!/bin/bash
conda activate tfnet

for file in results/*
do
    if [ -e $file/results.pt ]
    then
        if ! [ -e $file/results_val.pt ]
        then
            echo $file
            python TF-net/run_model.py --epoch 0 --only_val --d_ids 4 --path $file/
        fi
    fi
done