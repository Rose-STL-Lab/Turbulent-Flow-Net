#!/bin/bash

conda activate tfnet
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python TF_net/bayesian_opt.py 2>&1 | tee scripts/bayesian_opt/log.txt