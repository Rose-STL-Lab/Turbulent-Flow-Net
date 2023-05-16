#!/bin/bash

conda activate tfnet
export CUBLAS_WORKSPACE_CONFIG=:4096:8
data=data9_101
task=lyapunov
server=north
suffix=
rootdir=scripts_${server}/bayesian_opt/tfnet_${data}_${task}${suffix}/
mkdir -p $rootdir
max_mse=0.71
python TF_net/bayesian_opt.py --rootdir ${rootdir} --task ${task} --data ${data} --max_mse ${max_mse} 2>&1 | tee ${rootdir}log.txt