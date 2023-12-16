#!/bin/bash

conda activate tfnet
export CUBLAS_WORKSPACE_CONFIG=:4096:8
d_id=0
coef2=0
server=north
outln=15

array=( "17" "19" "41" "43" )
array2=( "4" "5" "6" "7" )
for pos_emb_dim in 4;  do
    for beta in 0.6 0.7 0.8 0.9; do
        for i in "${!array[@]}"; do
            seed="${array[i]}" 
            d_id="${array2[i]}"
            echo $seed $d_id
            name=${server}_tfnet_pos_emb_${pos_emb_dim}_outln_${outln}_beta_${beta}
            folder=${name}/${name}_${seed}
            mkdir -p results/$folder
            cp -v ${BASH_SOURCE[0]} results/$folder/
            python TF_net/run_model.py  --output_length ${outln} --beta ${beta} --pos_emb_dim $pos_emb_dim --desc $name --coef 0 --coef2 $coef2 --seed ${seed} --d_ids $d_id --path results/$folder/ \
                        2>&1 | tee results/$folder/log.txt &
        done
        wait
    done
done

wait

return
kill $(jobs -p)
# text=""
# --desc "$text"
#        name=tfnet_${data}_offset${dnsn}_noise_${noise}_wt_d_${wt_d}_enc_${addon_enc}-------add them to the comman also