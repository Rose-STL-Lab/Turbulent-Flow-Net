#!/bin/bash

conda activate tfnet
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# m_learnt
data=data21_101
slope=150
mixed_indx=
mixed_indx_no_overlap=
version="_3"

array=( "17" "19" "41" "43" )
array2=( "4" "4" "4" "4" )
for m_init in 0.4 0.3 0.2 0.1; do
    for i in "${!array[@]}"; do
        seed="${array[i]}" 
        d_id="${array2[i]}"
        echo $seed $d_id $m_init
        name=lya_${data}_coef2_1_m_learnt_${m_init}_s_${slope}
        folder=${name}/${name}_${seed}
        mkdir -p results/$folder
        python TF_net/run_model.py --version $version $mixed_indx $mixed_indx_no_overlap --epoch 0 --desc $name --slope ${slope} --m_init ${m_init} --data ${data}.pt --seed ${seed} --d_ids $d_id \
                        --path results/$folder/ 2>&1 | tee results/$folder/log.txt &
        wait
    done
    wait
done

# kill $(jobs -p)
# text=""
# --desc "$text"
#        name=lya_${data}_coef2_1_m_learnt_${m_init}_offset${dnsn}_noise_${noise}_wt_d_${wt_d}_enc_${addon_enc}_s_${slope}-------add them to the comman also