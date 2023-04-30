#!/bin/bash

conda activate tfnet
export CUBLAS_WORKSPACE_CONFIG=:4096:8
data=data9_101
task=norm_loss
server=north
rootdir=scripts_${server}/bayesian_opt/tfnet_${data}_${task}/
mkdir -p $rootdir
python TF_net/bayesian_opt.py --rootdir ${rootdir} --task ${task} 2>&1 | tee ${rootdir}log.txt