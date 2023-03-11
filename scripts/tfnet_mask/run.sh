#!/bin/bash

conda activate tfnet
export CUBLAS_WORKSPACE_CONFIG=:4096:8
d_id=0
coef2=0
mask=--mask
mstart=15
mend=50
mlower=80
mupper=100
epoch=100
mtile=16

array=( "19" "43" "17" "41" "53")
array2=( "1" "2" "4" "5" "6" )
for mstart in 15; do
    for mend in 40; do 
        if [ $mstart -eq 15 ] && [ $mend -eq 50 ]; then
            continue
        fi
        for i in "${!array[@]}"; do
            seed="${array[i]}" 
            d_id="${array2[i]}"
            echo $seed $d_id $mstart $mend $mtile
            name=tfnet_mask_${mstart}_${mend}_${mlower}_${mupper}_${epoch}_${mtile}
            folder=${name}/${name}_${seed}
            mkdir -p results/$folder
            cp -v ${BASH_SOURCE[0]} results/$folder/
            python TF_net/run_model.py $mask --desc $name --coef 0 --coef2 $coef2 --seed ${seed} --d_ids $d_id --path results/$folder/ \
                        --mstart $mstart --mend $mend --mend $mend --mupper $mupper --epoch $epoch --mtile $mtile \
                        2>&1 | tee results/$folder/log.txt &
        done
        wait
    done
    wait
done

wait

# kill $(jobs -p)
# text=""
# --desc "$text"