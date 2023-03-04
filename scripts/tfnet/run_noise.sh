#!/bin/bash

conda activate tfnet
export CUBLAS_WORKSPACE_CONFIG=:4096:8
d_id=0
coef2=0
not_use_test_mode=
noise=4
dnsn=--dnsn

array=( "17" "41" "47" "53")
array2=( "4" "5" "6" "7" )
for temp in 1 ; do
    for i in "${!array[@]}"; do
        seed="${array[i]}" 
        d_id="${array2[i]}"
        echo $seed $d_id $noise
        name=tfnet_${dnsn}_noise_${noise}
        folder=${name}/${name}_${seed}
        mkdir -p results/$folder
        cp -v ${BASH_SOURCE[0]} results/$folder/
        python TF_net/run_model.py $dnsn --noise $noise  --desc $name $not_use_test_mode --coef 0 --coef2 $coef2 --seed ${seed} --d_ids $d_id --path results/$folder/ \
                    2>&1 | tee results/$folder/log.txt &
    done
    wait
done

wait

# kill $(jobs -p)
# text=""
# --desc "$text"
#        name=tfnet_${data}_offset${dnsn}_noise_${noise}_wt_d_${wt_d}_enc_${addon_enc}-------add them to the comman also