#!/bin/bash

conda activate tfnet
export CUBLAS_WORKSPACE_CONFIG=:4096:8

d_id=0
coef2=0
data=data9_101
server=south
norm_loss=--norm_loss
outln=6
lr=0.0005
gamma=0.9304
epoch=120

array=( "17" "19" "41" "43" )
array2=( "2" "3" "5" "7" )
for temp in -1 ; do
    for temp2 in -1; do
        for i in "${!array[@]}"; do
            seed="${array[i]}" 
            d_id="${array2[i]}"
            echo $seed $d_id $trunc_factor
            name=${server}_tfnet_${data}${norm_loss}_lr_${lr}_gamma_${gamma}_epoch_${epoch}_outln_${outln}
            folder=${name}/${name}_${seed}
            mkdir -p results/$folder
            cp -v ${BASH_SOURCE[0]} results/$folder/
            python TF_net/run_model.py ${norm_loss} --data ${data}.pt --learning_rate ${lr} --gamma ${gamma} --output_length $outln --epoch $epoch \
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