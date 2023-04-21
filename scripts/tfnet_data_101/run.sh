 #!/bin/bash

conda activate tfnet
export CUBLAS_WORKSPACE_CONFIG=:4096:8
coef2=0
data=data9_101
mixed_indx=
mixed_indx_no_overlap=
version=""

array=( "17" "19" "41" "43" )
array2=( "4" "4" "4" "4" )
for outln in 5 6 7 8; do
    for i in "${!array[@]}"; do
        seed="${array[i]}" 
        d_id="${array2[i]}"
        echo $seed $d_id
        name=tfnet_${data}_outln_${outln}
        folder=${name}/${name}_${seed}
        mkdir -p results/$folder
        cp -v ${BASH_SOURCE[0]} results/$folder/
        python TF_net/run_model.py --output_length ${outln} --version $version  $mixed_indx $mixed_indx_no_overlap --epoch 0 --desc $name --data ${data}.pt --coef 0 --coef2 $coef2 --seed ${seed} --d_ids $d_id --path results/$folder/ \
                    2>&1 | tee results/$folder/log.txt &
        wait
    done
    wait
done

wait

# kill $(jobs -p)
# text=""
# --desc "$text"
#        name=tfnet_${data}_offset${dnsn}_noise_${noise}_wt_d_${wt_d}_enc_${addon_enc}-------add them to the comman also