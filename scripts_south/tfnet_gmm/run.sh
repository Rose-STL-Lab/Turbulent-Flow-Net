#!/bin/bash

conda activate tfnet
export CUBLAS_WORKSPACE_CONFIG=:4096:8
d_id=0
coef2=0
server=south
gmm_comp=3

array=( "17" "19" "41" "43" )
array2=( "2" "5" "6" "7" )
for lr in 2e-3 3e-3 4e-3 ; do
    for gamma in 0.9; do
        for i in "${!array[@]}"; do
            seed="${array[i]}" 
            d_id="${array2[i]}"
            echo $seed $d_id $trunc_factor
            name=${server}_tfnet_gmm_comp_${gmm_comp}_lr_${lr}_gamma_${gamma}
            folder=${name}/${name}_${seed}
            mkdir -p results/$folder
            cp -v ${BASH_SOURCE[0]} results/$folder/
            python TF_net/run_model.py --gmm_comp ${gmm_comp} --learning_rate ${lr} --gamma ${gamma} \
                --desc $name --coef 0 --coef2 $coef2 --seed ${seed} --d_ids $d_id --path results/$folder/ \
                        2>&1 | tee results/$folder/log.txt &
        done
        wait
    done
    wait
done

return
kill $(jobs -p)
# text=""
# --desc "$text"
#        name=tfnet_${data}_offset${dnsn}_noise_${noise}_wt_d_${wt_d}_enc_${addon_enc}-------add them to the comman also