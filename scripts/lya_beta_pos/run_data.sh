 #!/bin/bash

conda activate tfnet
export CUBLAS_WORKSPACE_CONFIG=:4096:8

slope=150
server=north
coef2=1

data=data9_101

m_val=0.3473
m_str_name=m_learnt
m_str_args="--m_init $m_val"

outln=6
beta=1

pos_emb_dim=14

inp_only=
name_inp_only=

# array=( "17" "19" "41" "43" )
# array2=( "2" "3" "2" "3" )
# for pos_emb_dim in 12 14 16 20; do
#     for i in "${!array[@]}"; do
#         seed="${array[i]}" 
#         d_id="${array2[i]}"
#         echo $seed $d_id
#         name=${server}_lya_${data}_coef2_${coef2}_${m_str_name}_${m_val}_outln_${outln}_beta_${beta}_pos_emb_${pos_emb_dim}${name_inp_only}
#         folder=${name}/${name}_${seed}
#         mkdir -p results/$folder
#         cp -v ${BASH_SOURCE[0]} results/$folder/
#         python TF_net/run_model.py --pos_emb_dim $pos_emb_dim $inp_only --output_length ${outln} --beta ${beta} --coef2 ${coef2} --desc ${name} --slope ${slope} \
#                         ${m_str_args} --data ${data}.pt --seed ${seed} --d_ids ${d_id} \
#                         --path results/${folder}/ 2>&1 | tee results/${folder}/log.txt &
#     done
#     wait
# done
# wait

array=( "17" "19" "41" "43" )
array2=( "0" "1" "0" "1" )
for pos_emb_dim in 12; do
    for i in "${!array[@]}"; do
        seed="${array[i]}" 
        d_id="${array2[i]}"
        echo $seed $d_id
        name=${server}_lya_${data}_coef2_${coef2}_${m_str_name}_${m_val}_outln_${outln}_beta_${beta}_pos_emb_${pos_emb_dim}${name_inp_only}
        folder=${name}/${name}_${seed}
        mkdir -p results/$folder
        cp -v ${BASH_SOURCE[0]} results/$folder/
        python TF_net/run_model.py --pos_emb_dim $pos_emb_dim $inp_only --output_length ${outln} --beta ${beta} --coef2 ${coef2} --desc ${name} --slope ${slope} \
                        ${m_str_args} --data ${data}.pt --seed ${seed} --d_ids ${d_id} \
                        --path results/${folder}/ 2>&1 | tee results/${folder}/log.txt &
    done
done

array=( "17" "19" "41" "43" )
array2=( "2" "3" "2" "3" )
for pos_emb_dim in 14; do
    for i in "${!array[@]}"; do
        seed="${array[i]}" 
        d_id="${array2[i]}"
        echo $seed $d_id
        name=${server}_lya_${data}_coef2_${coef2}_${m_str_name}_${m_val}_outln_${outln}_beta_${beta}_pos_emb_${pos_emb_dim}${name_inp_only}
        folder=${name}/${name}_${seed}
        mkdir -p results/$folder
        cp -v ${BASH_SOURCE[0]} results/$folder/
        python TF_net/run_model.py --pos_emb_dim $pos_emb_dim $inp_only --output_length ${outln} --beta ${beta} --coef2 ${coef2} --desc ${name} --slope ${slope} \
                        ${m_str_args} --data ${data}.pt --seed ${seed} --d_ids ${d_id} \
                        --path results/${folder}/ 2>&1 | tee results/${folder}/log.txt &
    done
done

array=( "17" "19" "41" "43" )
array2=( "4" "5" "4" "5" )
for pos_emb_dim in 16; do
    for i in "${!array[@]}"; do
        seed="${array[i]}" 
        d_id="${array2[i]}"
        echo $seed $d_id
        name=${server}_lya_${data}_coef2_${coef2}_${m_str_name}_${m_val}_outln_${outln}_beta_${beta}_pos_emb_${pos_emb_dim}${name_inp_only}
        folder=${name}/${name}_${seed}
        mkdir -p results/$folder
        cp -v ${BASH_SOURCE[0]} results/$folder/
        python TF_net/run_model.py --pos_emb_dim $pos_emb_dim $inp_only --output_length ${outln} --beta ${beta} --coef2 ${coef2} --desc ${name} --slope ${slope} \
                        ${m_str_args} --data ${data}.pt --seed ${seed} --d_ids ${d_id} \
                        --path results/${folder}/ 2>&1 | tee results/${folder}/log.txt &
    done
done

array=( "17" "19" "41" "43" )
array2=( "6" "7" "6" "7" )
for pos_emb_dim in 20; do
    for i in "${!array[@]}"; do
        seed="${array[i]}" 
        d_id="${array2[i]}"
        echo $seed $d_id
        name=${server}_lya_${data}_coef2_${coef2}_${m_str_name}_${m_val}_outln_${outln}_beta_${beta}_pos_emb_${pos_emb_dim}${name_inp_only}
        folder=${name}/${name}_${seed}
        mkdir -p results/$folder
        cp -v ${BASH_SOURCE[0]} results/$folder/
        python TF_net/run_model.py --pos_emb_dim $pos_emb_dim $inp_only --output_length ${outln} --beta ${beta} --coef2 ${coef2} --desc ${name} --slope ${slope} \
                        ${m_str_args} --data ${data}.pt --seed ${seed} --d_ids ${d_id} \
                        --path results/${folder}/ 2>&1 | tee results/${folder}/log.txt &
    done
done


wait


array=( "17" "19" "41" "43" )
array2=( "0" "1" "0" "1" )
for pos_emb_dim in 4; do
    for i in "${!array[@]}"; do
        seed="${array[i]}" 
        d_id="${array2[i]}"
        echo $seed $d_id
        name=${server}_lya_${data}_coef2_${coef2}_${m_str_name}_${m_val}_outln_${outln}_beta_${beta}_pos_emb_${pos_emb_dim}${name_inp_only}
        folder=${name}/${name}_${seed}
        mkdir -p results/$folder
        cp -v ${BASH_SOURCE[0]} results/$folder/
        python TF_net/run_model.py --pos_emb_dim $pos_emb_dim $inp_only --output_length ${outln} --beta ${beta} --coef2 ${coef2} --desc ${name} --slope ${slope} \
                        ${m_str_args} --data ${data}.pt --seed ${seed} --d_ids ${d_id} \
                        --path results/${folder}/ 2>&1 | tee results/${folder}/log.txt &
    done
done

array=( "17" "19" "41" "43" )
array2=( "2" "3" "2" "3" )
for pos_emb_dim in 6; do
    for i in "${!array[@]}"; do
        seed="${array[i]}" 
        d_id="${array2[i]}"
        echo $seed $d_id
        name=${server}_lya_${data}_coef2_${coef2}_${m_str_name}_${m_val}_outln_${outln}_beta_${beta}_pos_emb_${pos_emb_dim}${name_inp_only}
        folder=${name}/${name}_${seed}
        mkdir -p results/$folder
        cp -v ${BASH_SOURCE[0]} results/$folder/
        python TF_net/run_model.py --pos_emb_dim $pos_emb_dim $inp_only --output_length ${outln} --beta ${beta} --coef2 ${coef2} --desc ${name} --slope ${slope} \
                        ${m_str_args} --data ${data}.pt --seed ${seed} --d_ids ${d_id} \
                        --path results/${folder}/ 2>&1 | tee results/${folder}/log.txt &
    done
done

array=( "17" "19" "41" "43" )
array2=( "4" "5" "4" "5" )
for pos_emb_dim in 8; do
    for i in "${!array[@]}"; do
        seed="${array[i]}" 
        d_id="${array2[i]}"
        echo $seed $d_id
        name=${server}_lya_${data}_coef2_${coef2}_${m_str_name}_${m_val}_outln_${outln}_beta_${beta}_pos_emb_${pos_emb_dim}${name_inp_only}
        folder=${name}/${name}_${seed}
        mkdir -p results/$folder
        cp -v ${BASH_SOURCE[0]} results/$folder/
        python TF_net/run_model.py --pos_emb_dim $pos_emb_dim $inp_only --output_length ${outln} --beta ${beta} --coef2 ${coef2} --desc ${name} --slope ${slope} \
                        ${m_str_args} --data ${data}.pt --seed ${seed} --d_ids ${d_id} \
                        --path results/${folder}/ 2>&1 | tee results/${folder}/log.txt &
    done
done

array=( "17" "19" "41" "43" )
array2=( "6" "7" "6" "7" )
for pos_emb_dim in 24; do
    for i in "${!array[@]}"; do
        seed="${array[i]}" 
        d_id="${array2[i]}"
        echo $seed $d_id
        name=${server}_lya_${data}_coef2_${coef2}_${m_str_name}_${m_val}_outln_${outln}_beta_${beta}_pos_emb_${pos_emb_dim}${name_inp_only}
        folder=${name}/${name}_${seed}
        mkdir -p results/$folder
        cp -v ${BASH_SOURCE[0]} results/$folder/
        python TF_net/run_model.py --pos_emb_dim $pos_emb_dim $inp_only --output_length ${outln} --beta ${beta} --coef2 ${coef2} --desc ${name} --slope ${slope} \
                        ${m_str_args} --data ${data}.pt --seed ${seed} --d_ids ${d_id} \
                        --path results/${folder}/ 2>&1 | tee results/${folder}/log.txt &
    done
done

# kill $(jobs -p)
# text=""
# --desc "$text"
#        name=tfnet_${data}_offset${dnsn}_noise_${noise}_wt_d_${wt_d}_enc_${addon_enc}-------add them to the comman also