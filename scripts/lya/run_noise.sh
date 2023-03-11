#!/bin/bash

conda activate tfnet
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# m_learnt
dnsn=--dnsn
noise=4
m_init=0.09

array=( "17" "19" "41" "43" "47" "53")
array2=( "2" "3" "4" "5" "6" "7" )
for slope in 150; do
    for i in "${!array[@]}"; do
        seed="${array[i]}" 
        d_id="${array2[i]}"
        echo $seed $d_id $slope
        name=lya_coef2_1_m_learnt_${m_init}_offset${dnsn}_noise_${noise}_s_${slope}
        folder=${name}/${name}_${seed}
        mkdir -p results/$folder
        python TF_net/run_model.py --slope ${slope} --noise ${noise} $dnsn --m_init ${m_init} --seed ${seed} --d_ids $d_id \
                        --path results/$folder/ 2>&1 | tee results/$folder/log.txt &
    done
    wait
done

# kill $(jobs -p)
# text=""
# --desc "$text"