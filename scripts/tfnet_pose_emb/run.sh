#!/bin/bash

conda activate tfnet
export CUBLAS_WORKSPACE_CONFIG=:4096:8
d_id=0
coef2=0
pos_emb_dim=64
outln=6

array=( "17" "19" "41" "43" "53" )
array2=( "3" "1" "2" "5" "6" )
for dfasdf in 0.3 ;  do
    for i in "${!array[@]}"; do
        seed="${array[i]}" 
        d_id="${array2[i]}"
        echo $seed $d_id
        name=tfnet_pos_emb_${pos_emb_dim}_outln_${outln}
        folder=${name}/${name}_${seed}
        mkdir -p results/$folder
        cp -v ${BASH_SOURCE[0]} results/$folder/
        python TF_net/run_model.py --epoch 0 --output_length ${outln} --pos_emb_dim $pos_emb_dim --desc $name --coef 0 --coef2 $coef2 --seed ${seed} --d_ids $d_id --path results/$folder/ \
                    2>&1 | tee results/$folder/log.txt &
    done
    wait
done

wait

return
kill $(jobs -p)
# text=""
# --desc "$text"
#        name=tfnet_${data}_offset${dnsn}_noise_${noise}_wt_d_${wt_d}_enc_${addon_enc}-------add them to the comman also