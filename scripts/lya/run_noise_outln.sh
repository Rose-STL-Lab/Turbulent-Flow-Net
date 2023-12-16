#!/bin/bash

conda activate tfnet
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# m_learnt
data=data21_101
dnsn=--dnsn
noise=4
slope=150
m_init=0.4

array=( "17" "19" "41" "43" )
array2=( "0" "1" "2" "3" )
for outln in 5 6; do
    for i in "${!array[@]}"; do
        seed="${array[i]}" 
        d_id="${array2[i]}"
        echo $seed $d_id $addon_enc $outln
        name=lya_${data}_coef2_1_outln_${outln}_m_learnt_${m_init}_s_${slope}${dnsn}_noise_${noise}
        folder=${name}/${name}_${seed}
        mkdir -p results/$folder
        python TF_net/run_model.py --output_length $outln ${dnsn} --noise ${noise} --slope ${slope} --m_init ${m_init} --data ${data}.pt --seed ${seed} --d_ids $d_id \
                        --path results/$folder/ 2>&1 | tee results/$folder/log.txt &
    done
    wait
done

# kill $(jobs -p)
# text=""
# --desc "$text"
#        name=lya_${data}_coef2_1_m_learnt_${m_init}_offset${dnsn}_noise_${noise}_wt_d_${wt_d}_enc_${addon_enc}_s_${slope}-------add them to the comman also