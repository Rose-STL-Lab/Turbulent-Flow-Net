 #!/bin/bash

conda activate tfnet
export CUBLAS_WORKSPACE_CONFIG=:4096:8
coef2=0
data=data5
dnsn=--dnsn
noise=4
wt_d=1e-5
addon_enc=3

# array=( "19" "97" "7" "17" )
# array2=( "4" "5" "6" "7" )
array=( "17" )
array2=( "1" )
for temp in 1 ; do
    for i in "${!array[@]}"; do
        seed="${array[i]}" 
        d_id="${array2[i]}"
        echo $seed $d_id $noise
        name=tfnet_data5_offset${dnsn}_noise_${noise}_wt_d_${wt_d}_enc_${addon_enc}
        folder=${name}/${name}_${seed}
        mkdir -p results/$folder
        cp -v ${BASH_SOURCE[0]} results/$folder/
        python TF_net/run_model.py --desc $name --epoch 0 --addon_enc $addon_enc --wt_decay $wt_d --noise ${noise} $dnsn --data ${data}.pt --coef 0 --coef2 $coef2 --seed ${seed} --d_ids $d_id --path results/$folder/ \
                    2>&1 | tee results/$folder/log.txt &
    done
    wait
done

wait

# kill $(jobs -p)
# text=""
# --desc "$text"