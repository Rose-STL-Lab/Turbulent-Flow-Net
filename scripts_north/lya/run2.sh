#!/bin/bash

conda activate tfnet
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# m_learnt
data=data5
dnsn=--dnsn
noise=4
wt_d=1e-5
m_init=0.09
addon_enc=3

array=( "19" "97" "7" "17" )
array2=( "4" "5" "6" "7" )
for slope in 150; do
    for i in "${!array[@]}"; do
        seed="${array[i]}" 
        d_id="${array2[i]}"
        echo $seed $d_id $addon_enc
        name=lya_${data}_coef2_1_m_learnt_${m_init}_offset${dnsn}_noise_${noise}_wt_d_${wt_d}_enc_${addon_enc}_s_${slope}
        folder=${name}/${name}_${seed}
        mkdir -p results/$folder
        python TF_net/run_model.py  --desc $name --slope ${slope} --addon_enc $addon_enc --wt_decay $wt_d --noise ${noise} $dnsn --m_init ${m_init} --data ${data}.pt --seed ${seed} --d_ids $d_id \
                        --path results/$folder/ 2>&1 | tee results/$folder/log.txt &
    done
    wait
done

# kill $(jobs -p)
# text=""
# --desc "$text"