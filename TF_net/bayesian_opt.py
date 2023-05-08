# %%
from ax.service.managed_loop import optimize
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.utils.notebook.plotting import render
import torch
import numpy as np
import glob
import subprocess
from subprocess import PIPE, STDOUT
import pandas as pd
import os
import time
import pickle
import argparse
from functools import partial

def load_and_delete_pred(f):
    temp = torch.load(f)
    torch.save({"loss_curve": temp['loss_curve']}, f, pickle_protocol=5)
    return temp['loss_curve']

def get_files(files, thresh):
    if type(files) != list:
        files = [files]
    if not len(files):
        raise ValueError("Empty files")
    d = []
    _len=0
    for i, f in enumerate(files):
        temp = load_and_delete_pred(f)
        _len = len(temp)
        if temp[-1]>thresh:
            print(f"dropping {f}, diverged!")
        else:
            print("done ", f)
            d.append(temp)
    d = np.vstack(d) if len(d) else np.array([[0]*_len])
    return d.mean(axis=0), d.std(axis=0)

# %%
def evaluate_mask_opt(params, data):

    # Generate Results
    print(f"Running with params: {params}")
    d_id=0
    coef2=0
    server="south"
    mask="--mask"
    mstart=0
    mend=params['mend']
    mlower=round(params['mlower'],2)
    lr=params['lr']
    gamma=params['gamma']
    mupper=100
    epoch=100
    mtile=1
    inplen=32
    mtype="opt"
    seed_arr=( "19","43","17","41")
    d_id_arr=( "1","2","4","0" )
    name=f"{server}_tfnet_{data}_mask_{mtype}_{mstart}_{mend}_{mlower}_{mupper}_{epoch}_{mtile}_inplen_{inplen}_lr_{lr}_gamma_{gamma}"

    ps=[]
    for seed, d_id in zip(seed_arr, d_id_arr):
        print(f"Running seed:{seed}, on d_id:{d_id}", flush=True)
        folder=f"{name}/{name}_{seed}"
        print(f"folder:{folder}, seed: {seed}", flush=True)
        os.makedirs(f"results/{folder}", exist_ok=True)
        cmd = f"(python TF_net/run_model.py --learning_rate {lr} --gamma {gamma} --num_workers 1 --input_length {inplen} --data {data}.pt --desc {name} --coef 0 --coef2 {coef2} --seed {seed} --d_ids {d_id} --path results/{folder}/ \
                         {mask} --mtype {mtype} --mstart {mstart} --mend {mend} --mlower {mlower} --mupper {mupper} --epoch {epoch} --mtile {mtile} \
                        2>&1 | tee results/{folder}/log.txt)"
        ps.append(subprocess.Popen(cmd, shell=True, close_fds=True, executable="/bin/bash"))
    for i,p in enumerate(ps):
        p.communicate()
        print(f"seed:{seed_arr[i]} returned !", flush=True)

    # Get results
    seeds = [43,41,17,19]
    beta = 2.3853
    thresh = float('inf')
    test='_val'

    check_seed = lambda x, _seeds: -1 in seeds or (int(x.rsplit("_", 1)[1]) in _seeds)
    v = list(glob.glob("./results/" + name + "/*"))
    v = [i + f"/results{test}.pt" for i in v if check_seed(i, seeds)]
    results = get_files(v, thresh)
    return {'mse': (results[0][-1], results[1][-1])}

def evaluate_norm_loss(params, data):

    # Generate Results
    print(f"Running with params: {params}")
    d_id=0
    coef2=0
    server="south"
    norm_loss="--norm_loss"
    lr=params['lr']
    gamma=round(params['gamma'],4)
    epoch=120
    seed_arr=( "19","43","17","41")
    d_id_arr=( "7","1","4","6" )
    name=f"{server}_tfnet_{data}{norm_loss}_lr_{lr}_gamma_{gamma}_epoch_{epoch}"

    ps=[]
    for seed, d_id in zip(seed_arr, d_id_arr):
        print(f"Running seed:{seed}, on d_id:{d_id}", flush=True)
        folder=f"{name}/{name}_{seed}"
        print(f"folder:{folder}, seed: {seed}", flush=True)
        os.makedirs(f"results/{folder}", exist_ok=True)
        cmd = f"(python TF_net/run_model.py {norm_loss} --epoch {epoch} --data {data}.pt --learning_rate {lr} --gamma {gamma} \
                --desc {name} --coef 0 --coef2 {coef2} --seed {seed} --d_ids {d_id} --path results/{folder}/ \
                        2>&1 | tee results/{folder}/log.txt)"
        ps.append(subprocess.Popen(cmd, shell=True, close_fds=True, executable="/bin/bash"))
    for i,p in enumerate(ps):
        p.communicate()
        print(f"seed:{seed_arr[i]} returned !", flush=True)

    # Get results
    seeds = [43,41,17,19]
    beta = 2.3853
    thresh = float('inf')
    test='_val'

    check_seed = lambda x, _seeds: -1 in seeds or (int(x.rsplit("_", 1)[1]) in _seeds)
    v = list(glob.glob("./results/" + name + "/*"))
    v = [i + f"/results{test}.pt" for i in v if check_seed(i, seeds)]
    results = get_files(v, thresh)
    return {'mse': (results[0][-1], results[1][-1])}

def evaluate_lyapunov(params, data, max_mse=4.0):
    # Generate Results
    print(f"Running with params: {params}")
    params['m_val'] = round(params['m_val'], 4)
    d_id=0
    coef2=1
    server="north"
    slope=150
    if params['m_learnt']:
        m_str_name='m_learnt'
        m_str_args=f"--m_init {params['m_val']}"
    else:
        m_str_name='m'
        m_str_args=f"--mide {params['m_val']}"
    seed_arr=( "19","43","17","41")
    d_id_arr=( "0","1","4","6" )
    name=f"{server}_lya_{data}_coef2_{coef2}_{m_str_name}_{params['m_val']}_s_{slope}"

    ps=[]
    for seed, d_id in zip(seed_arr, d_id_arr):
        print(f"Running seed:{seed}, on d_id:{d_id}", flush=True)
        folder=f"{name}/{name}_{seed}"
        print(f"folder:{folder}, seed: {seed}", flush=True)
        os.makedirs(f"results/{folder}", exist_ok=True)
        cmd = f"(python TF_net/run_model.py --coef2 {coef2} --desc {name} --slope {slope} {m_str_args} --data {data}.pt --seed {seed} --d_ids {d_id} \
                        --path results/{folder}/ 2>&1 | tee results/{folder}/log.txt)"
        ps.append(subprocess.Popen(cmd, shell=True, close_fds=True, executable="/bin/bash"))
    for i,p in enumerate(ps):
        p.communicate()
        print(f"seed:{seed_arr[i]} returned !", flush=True)

    # Get results
    seeds = [43,41,17,19]
    beta = 2.3853
    thresh = float('inf')
    test='_val'
    try:
        check_seed = lambda x, _seeds: -1 in seeds or (int(x.rsplit("_", 1)[1]) in _seeds)
        v = list(glob.glob("./results/" + name + "/*"))
        v = [i + f"/results{test}.pt" for i in v if check_seed(i, seeds)]
        results = get_files(v, thresh)
        return {'mse': (results[0][-1], results[1][-1])}
    except FileNotFoundError:
        return {'mse': (max_mse, 0.00001)}

# %%
# Ref: https://ax.dev/api/_modules/ax/service/ax_client.html#AxClient.create_experiment
ax_client = AxClient()
ax_client.create_experiment(
    name="Lyapunov",
    parameters=[
        {
            "name": "m_learnt",
            "type": "choice",
            "values": [True, False],
            "value_type": "bool",  # Optional, defaults to inference from type of "bounds".
            "is_ordered": False,
        },
        {
            "name": "m_val",
            "type": "range",
            "bounds": [0.2, 0.6],
            "value_type": "float",  # Optional, defaults to inference from type of "bounds".
            "log_scale": False,  # Optional, defaults to False.
        },
    ],
    objectives={"mse": ObjectiveProperties(minimize=True)}  # This mse is just a reference name to return outputs to.
)

parser = argparse.ArgumentParser()
parser.add_argument("--rootdir",
                    type=str)
parser.add_argument("--task",
                    type=str)
parser.add_argument("--data",
                    type=str)
parser.add_argument("--max_mse",
                    type=float,
                    help="default mse if run fails, used for lyapunov")


args= parser.parse_args()
rootdir=args.rootdir
task=args.task

if task not in ['mask_opt', 'norm_loss', 'lyapunov']:
    raise ValueError("task not recognizied, check spelling!")
if task == 'mask_opt':
    evaluate = evaluate_mask_opt
elif task == 'norm_loss':
    evaluate = evaluate_norm_loss
elif task == 'lyapunov':
    evaluate = partial(evaluate_lyapunov, max_mse=args.max_mse)
else:
    raise ValueError("Shouldn't happen!")
evaluate = partial(evaluate, data=args.data)

# evaluate({'m_learnt': True, 'm_val': 0.4})

os.makedirs(rootdir, exist_ok=True)
results=[]
for i in range(50):
    parameters, trial_index = ax_client.get_next_trial()
    # Local evaluation here can be replaced with deployment to external system.
    raw_data=evaluate(parameters)
    ax_client.complete_trial(trial_index=trial_index, raw_data=raw_data)

    parameters['raw_data'] = raw_data
    results.append(parameters)

    print(ax_client.generation_strategy.trials_as_df)
    ax_client.generation_strategy.trials_as_df.to_csv(rootdir+"trials.csv")

    print(ax_client.get_best_parameters())
    with open(rootdir+"best_parameters.pickle", 'wb') as handle:
        pickle.dump(ax_client.get_best_parameters(), handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(rootdir+"results.pickle", 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

