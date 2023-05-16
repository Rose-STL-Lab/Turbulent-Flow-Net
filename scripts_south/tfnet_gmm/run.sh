#!/bin/bash

# outlen
# gmm_comp
# 

conda activate tfnet
export CUBLAS_WORKSPACE_CONFIG=:4096:8
d_id=0
coef2=0
server=south
outln=6
ignore_min2=--ignore_min2

array=( "17" "19" "41" "43" )
array2=( "2" "3" "4" "5" )
for gmm_comp in 5 4 ; do
    for ignore_min2_epoch in 12; do
        for i in "${!array[@]}"; do
            seed="${array[i]}" 
            d_id="${array2[i]}"
            echo $seed $d_id $trunc_factor
            name=${server}_tfnet_gmm_comp_${gmm_comp}${ignore_min2}_${ignore_min2_epoch}_outln_${outln}
            folder=${name}/${name}_${seed}
            mkdir -p results/$folder
            cp -v ${BASH_SOURCE[0]} results/$folder/
            python TF_net/run_model.py --output_length ${outln} --gmm_comp ${gmm_comp} ${ignore_min2} --ignore_min2_epoch ${ignore_min2_epoch}\
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