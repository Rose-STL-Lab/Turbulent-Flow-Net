 #!/bin/bash

conda activate tfnet
export CUBLAS_WORKSPACE_CONFIG=:4096:8

slope=150
server=north
coef2=1

m_val=0.4469
m_str_name=m
m_str_args="--mide $m_val"

outln=6
beta=0.5931

dnsn=--dnsn
noise=4

array=( "17" "19" "41" "43" )
array2=( "7" "4" "5" "6" )
for data in data20_101 ; do
    for pos_emb_dim in 20 14 12 8; do
        for i in "${!array[@]}"; do
            seed="${array[i]}" 
            d_id="${array2[i]}"
            echo $seed $d_id
            name=${server}_lya_${data}_coef2_${coef2}_${m_str_name}_${m_val}_outln_${outln}_beta_${beta}_pos_emb_${pos_emb_dim}${dnsn}_noise_${noise}
            folder=${name}/${name}_${seed}
            mkdir -p results/$folder
            cp -v ${BASH_SOURCE[0]} results/$folder/
            python TF_net/run_model.py --pos_emb_dim $pos_emb_dim ${dnsn} --noise ${noise} --output_length ${outln} --beta ${beta} --coef2 ${coef2} --desc ${name} --slope ${slope} \
                            ${m_str_args} --data ${data}.pt --seed ${seed} --d_ids ${d_id} \
                            --path results/${folder}/ 2>&1 | tee results/${folder}/log.txt &
        done
        wait
    done
    wait
done

# kill $(jobs -p)
# text=""
# --desc "$text"
#        name=tfnet_${data}_offset${dnsn}_noise_${noise}_wt_d_${wt_d}_enc_${addon_enc}-------add them to the comman also