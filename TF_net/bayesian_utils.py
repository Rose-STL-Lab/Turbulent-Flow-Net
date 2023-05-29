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

# Ref: https://ax.dev/api/_modules/ax/service/ax_client.html#AxClient.create_experiment
lr_experiment = {
    'name':"lr",
    'parameters':[
        {
            "name": "lr",
            "type": "range",
            "bounds": [5e-4, 2e-3],
            "value_type": "float",  # Optional, defaults to inference from type of "bounds".
            "log_scale": False,  # Optional, defaults to False.
        },
        {
            "name": "gamma",
            "type": "range",
            "bounds": [0.9, 0.96],
            "value_type": "float",  # Optional, defaults to inference from type of "bounds".
            "log_scale": False,  # Optional, defaults to False.
        },
    ],
    'objectives':{"mse": ObjectiveProperties(minimize=True)}  # This mse is just a reference name to return outputs to.
}

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
    lr=round(params['lr'],8)
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
    thresh = float('inf')
    test='_val'

    check_seed = lambda x, _seeds: -1 in seeds or (int(x.rsplit("_", 1)[1]) in _seeds)
    v = list(glob.glob("./results/" + name + "/*"))
    v = [i + f"/results{test}.pt" for i in v if check_seed(i, seeds)]
    results = get_files(v, thresh)
    return {'mse': (results[0][-1], results[1][-1])}

def evaluate_gmm(params, data):
    # Generate Results
    print(f"Running with params: {params}")
    d_id=0
    coef2=0
    server="south"
    lr=round(params['lr'],8)
    gamma=round(params['gamma'],4)
    gmm_comp = params['gmm_comp']
    seed_arr=( "19","43","17","41")
    d_id_arr=( "2","5","4","7" )
    name=f"{server}_tfnet_{data}_gmm_comp_{gmm_comp}_lr_{lr}_gamma_{gamma}"

    ps=[]
    for seed, d_id in zip(seed_arr, d_id_arr):
        print(f"Running seed:{seed}, on d_id:{d_id}", flush=True)
        folder=f"{name}/{name}_{seed}"
        print(f"folder:{folder}, seed: {seed}", flush=True)
        os.makedirs(f"results/{folder}", exist_ok=True)
        cmd = f"(python TF_net/run_model.py --gmm_comp {gmm_comp} --data {data}.pt --learning_rate {lr} --gamma {gamma} \
                --desc {name} --coef 0 --coef2 {coef2} --seed {seed} --d_ids {d_id} --path results/{folder}/ \
                        2>&1 | tee results/{folder}/log.txt)"
        ps.append(subprocess.Popen(cmd, shell=True, close_fds=True, executable="/bin/bash"))
    for i,p in enumerate(ps):
        p.communicate()
        print(f"seed:{seed_arr[i]} returned !", flush=True)

    # Get results
    seeds = [43,41,17,19]
    thresh = float('inf')
    test='_val'

    check_seed = lambda x, _seeds: -1 in seeds or (int(x.rsplit("_", 1)[1]) in _seeds)
    v = list(glob.glob("./results/" + name + "/*"))
    v = [i + f"/results{test}.pt" for i in v if check_seed(i, seeds)]
    results = get_files(v, thresh)
    return {'mse': (results[0][-1], results[1][-1])}


#========================================Lyapunov Exp===================================================================

# Ref: https://ax.dev/api/_modules/ax/service/ax_client.html#AxClient.create_experiment
lyapunov_experiment = {
    'name':"Lyapunov",
    'parameters':[
        {
            "name": "gmm_comp",
            "type": "range",
            "bounds": [4, 7],
            "value_type": "int",  # Optional, defaults to inference from type of "bounds".
            "is_ordered": False,
        },
        {
            "name": "lr",
            "type": "range",
            "bounds": [5e-4, 2e-3],
            "value_type": "float",  # Optional, defaults to inference from type of "bounds".
            "log_scale": False,  # Optional, defaults to False.
        },
        {
            "name": "gamma",
            "type": "range",
            "bounds": [0.9, 0.96],
            "value_type": "float",  # Optional, defaults to inference from type of "bounds".
            "log_scale": False,  # Optional, defaults to False.
        },
    ],
    'objectives':{"mse": ObjectiveProperties(minimize=True)}  # This mse is just a reference name to return outputs to.
}

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
    d_id_arr=( "1","2","3","4" )
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
    thresh = float('inf')
    test='_val'
    try:
        check_seed = lambda x, _seeds: -1 in seeds or (int(x.rsplit("_", 1)[1]) in _seeds)
        v = list(glob.glob("./results/" + name + "/*"))
        v = [i + f"/results{test}.pt" for i in v if check_seed(i, seeds)]
        results = get_files(v, thresh)
        return {'mse': (results[0][-1], results[1][-1])}
    except FileNotFoundError:
        return {'mse': (max_mse, 0.01)}
    

#========================================dfusn===================================================================

# Ref: https://ax.dev/api/_modules/ax/service/ax_client.html#AxClient.create_experiment
dfusn_experiment = {
    'name':"dfusn",
    'parameters':[
        {
            "name": "dfusn_alpha",
            "type": "range",
            "bounds": [0.7, 0.99],
            "value_type": "float",  # Optional, defaults to inference from type of "bounds".
            "log_scale": False,  # Optional, defaults to False.
        },
        {
            "name": "wt_decay",
            "type": "range",
            "bounds": [4e-4, 1e-2],
            "value_type": "float",  # Optional, defaults to inference from type of "bounds".
            "log_scale": True,  # Optional, defaults to False.
        },
    ],
    'objectives':{"mse": ObjectiveProperties(minimize=True)}  # This mse is just a reference name to return outputs to.
}

def evaluate_dfusn(params, data, max_mse=0.472):
    # Generate Results
    print(f"Running with params: {params}")
    d_id=0
    coef2=0
    server="south"
    wt_decay = round(params['wt_decay'], 8)
    outln=6
    dfusn_alpha = round(params['dfusn_alpha'], 4)
    seed_arr=( "19","43","17","41")
    d_id_arr=( "3","4","5","7" )
    name=f"{server}_tfnet_{data}_outln_{outln}_dfusn_alpha_{dfusn_alpha}_wtdecay_{wt_decay}"

    ps=[]
    for seed, d_id in zip(seed_arr, d_id_arr):
        print(f"Running seed:{seed}, on d_id:{d_id}", flush=True)
        folder=f"{name}/{name}_{seed}"
        print(f"folder:{folder}, seed: {seed}", flush=True)
        os.makedirs(f"results/{folder}", exist_ok=True)
        cmd = f"(python TF_net/run_model.py --data {data}.pt --wt_decay {wt_decay} --output_length {outln} --dfusn_alpha {dfusn_alpha} --desc {name} --coef 0 --coef2 {coef2} --seed {seed} --d_ids {d_id} --path results/{folder}/ \
                        2>&1 | tee results/{folder}/log.txt)"
        ps.append(subprocess.Popen(cmd, shell=True, close_fds=True, executable="/bin/bash"))
    for i,p in enumerate(ps):
        p.communicate()
        print(f"seed:{seed_arr[i]} returned !", flush=True)

    # Get results
    seeds = [43,41,17,19]
    thresh = float('inf')
    test='_val'
    check_seed = lambda x, _seeds: -1 in seeds or (int(x.rsplit("_", 1)[1]) in _seeds)
    v = list(glob.glob("./results/" + name + "/*"))
    v = [i + f"/results{test}.pt" for i in v if check_seed(i, seeds)]
    results = get_files(v, thresh)
    return {'mse': (results[0][-1], results[1][-1])}