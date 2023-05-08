#!/bin/bash

conda activate tfnet
export CUBLAS_WORKSPACE_CONFIG=:4096:8

coef2=0
data=data9_101
mask=--mask
server=south
mstart=0
mend=70
mlower=74.88
mupper=100
epoch=100
mtile=16
inplen=32

array=( "19" "43" "17" "41" )
array2=( "0" "1" "2" "3" )
for sdfsdf in -1; do
    for dfsd in -1; do 
        name=${server}_tfnet_${data}_mask_${mstart}_${mend}_${mlower}_${mupper}_${epoch}_${mtile}_inplen_${inplen}
        for i in "${!array[@]}"; do
            seed="${array[i]}" 
            d_id="${array2[i]}"
            echo $seed $d_id $data
            folder=${name}/${name}_${seed}
            mkdir -p results/$folder
            cp -v ${BASH_SOURCE[0]} results/$folder/
            (python TF_net/run_model.py --input_length $inplen --data ${data}.pt --desc $name --coef 0 --coef2 $coef2 --seed ${seed} --d_ids $d_id --path results/$folder/ \
                         $mask --mstart $mstart --mend $mend --mlower $mlower --mupper $mupper --epoch $epoch --mtile $mtile \
                        2>&1 | tee results/$folder/log.txt) &
        done
        wait
    done
    wait
done

wait

# if [ $mstart -eq 15 ] && [ $mend -eq 50 ]; then
#     continue
# fi
# kill $(jobs -p)
# pkill -u lakshya -9 python
# text=""
# --desc "$text"