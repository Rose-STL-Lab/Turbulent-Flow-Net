#!/bin/bash

conda activate tfnet
export CUBLAS_WORKSPACE_CONFIG=:4096:8
d_id=0
coef2=0
not_use_test_mode=
input_length=27
array=( "17" "19" "41" "43" "47" "53")
array2=( "6" "1" "2" "3" "4" "5" )
for temp in 1 ; do
    for i in "${!array[@]}"; do
        seed="${array[i]}" 
        d_id="${array2[i]}"
        echo $seed $d_id $noise
        name=tfnet_mask_100
        folder=${name}/${name}_${seed}
        mkdir -p results/$folder
        cp -v ${BASH_SOURCE[0]} results/$folder/
        python TF_net/run_model.py $not_use_test_mode --epoch 0 --input_length $input_length --coef 0 --coef2 $coef2 --seed ${seed} --d_ids $d_id --path results/$folder/ \
                    2>&1 | tee results/$folder/log.txt &
    done
    wait
done

wait

# kill $(jobs -p)
# text=""
# --desc "$text"