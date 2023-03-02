#!/bin/bash

conda activate tfnet
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# m_learnt
data=data9_101
slope=150

array=( "17" "19" "41" "43" )
array2=( "4" "5" "6" "7" )
for outln in 8 ; do
    for m_init in 0.5 0.4 0.3; do
        for i in "${!array[@]}"; do
            seed="${array[i]}" 
            d_id="${array2[i]}"
            echo $seed $d_id $outln $m_init
            name=lya_${data}_coef2_1_outln_${outln}_m_learnt_${m_init}
            folder=${name}/${name}_${seed}
            mkdir -p results/$folder
            python TF_net/run_model.py --output_length $outln --slope ${slope} --m_init ${m_init} --data ${data}.pt --seed ${seed} --d_ids $d_id \
                            --path results/$folder/ 2>&1 | tee results/$folder/log.txt &
        done
        wait
    done
done

# kill $(jobs -p)
# text=""
# --desc "$text"