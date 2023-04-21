#!/bin/bash

conda activate tfnet
export CUBLAS_WORKSPACE_CONFIG=:4096:8
d_id=0
coef2=0
server=north
# trunc=4
# output_length=6
# trunc_factor=0.05
norm_loss=--norm_loss

array=( "17" "19" "41" "43" )
array2=( "1" "5" "6" "7" )
for lr in 1e-3 5e-2 1e-2 ; do
    for gamma in 0.94 0.9; do
        for i in "${!array[@]}"; do
            seed="${array[i]}" 
            d_id="${array2[i]}"
            echo $seed $d_id $trunc_factor
            # name=${server}_tfnet_trunc_${trunc}_${trunc_factor}_outln_${output_length}
            name=${server}_tfnet${norm_loss}_lr_${lr}_gamma_${gamma}
            folder=${name}/${name}_${seed}
            mkdir -p results/$folder
            cp -v ${BASH_SOURCE[0]} results/$folder/
            python TF_net/run_model.py ${norm_loss} --learning_rate ${lr} --gamma ${gamma}\
                # --output_length ${output_length} --trunc ${trunc} --trunc_factor ${trunc_factor} 
                --desc $name --coef 0 --coef2 $coef2 --seed ${seed} --d_ids $d_id --path results/$folder/ \
                        2>&1 | tee results/$folder/log.txt &
        done
        wait
    done
    wait
done

wait

return
kill $(jobs -p)
# text=""
# --desc "$text"
#        name=tfnet_${data}_offset${dnsn}_noise_${noise}_wt_d_${wt_d}_enc_${addon_enc}-------add them to the comman also