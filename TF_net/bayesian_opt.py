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
def evaluate(params):

    # Generate Results
    print(f"Running with params: {params}")
    d_id=0
    coef2=0
    data="data9_101"
    mask="--mask"
    mstart=params['mstart']
    mend=params['mend']
    mlower=round(params['mlower'],2)
    mupper=100
    epoch=100
    mtile=16
    inplen=32
    seed_arr=( "19","43","17","41","53")
    d_id_arr=( "1","2","4","5","6" )
    name=f"tfnet_{data}_mask_{mstart}_{mend}_{mlower}_{mupper}_{epoch}_{mtile}_inplen_{inplen}"

    ps=[]
    for seed, d_id in zip(seed_arr, d_id_arr):
        print(f"Running seed:{seed}, on d_id:{d_id}", flush=True)
        folder=f"{name}/{name}_{seed}"
        os.makedirs(f"results/{folder}", exist_ok=True)
        cmd = f"(python TF_net/run_model.py --num_workers 1 --input_length {inplen} --data {data}.pt --desc {name} --coef 0 --coef2 {coef2} --seed {seed} --d_ids {d_id} --path results/{folder}/ \
                         {mask} --mstart {mstart} --mend {mend} --mlower {mlower} --mupper {mupper} --epoch {epoch} --mtile {mtile} \
                        2>&1 | tee results/{folder}/log.txt)"
        ps.append(subprocess.Popen(cmd, shell=True, close_fds=True, executable="/bin/bash"))
    for i,p in enumerate(ps):
        p.communicate()
        print(f"seed:{seed_arr[i]} returned !", flush=True)

    # Get results
    seeds = [43,41,53,17,19]
    beta = 2.3853
    thresh = float('inf')
    test='_val'

    check_seed = lambda x, _seeds: -1 in seeds or (int(x.rsplit("_", 1)[1]) in _seeds)
    v = list(glob.glob("./results/" + name + "/*"))
    v = [i + f"/results{test}.pt" for i in v if check_seed(i, seeds)]
    results = get_files(v, thresh)
    return {'mse': (results[0][-1], results[1][-1])}

# %%
ax_client = AxClient()
ax_client.create_experiment(
    name="Lyapunov",
    parameters=[
        {
            "name": "mstart",
            "type": "range",
            "bounds": [0, 25],
            "value_type": "int",  # Optional, defaults to inference from type of "bounds".
            "log_scale": False,  # Optional, defaults to False.
        },
        {
            "name": "mend",
            "type": "range",
            "bounds": [40, 70],
        },
        {
            "name": "mlower",
            "type": "range",
            "bounds": [70.0,90.0],
        },
    ],
    objectives={"mse": ObjectiveProperties(minimize=True)}
)

# evaluate({'mstart':0, 'mend':60, 'mlower':79.0054100058})

results=[]
for i in range(50):
    parameters, trial_index = ax_client.get_next_trial()
    # Local evaluation here can be replaced with deployment to external system.
    raw_data=evaluate(parameters)
    ax_client.complete_trial(trial_index=trial_index, raw_data=raw_data)

    parameters['raw_data'] = raw_data
    results.append(parameters)

    print(ax_client.generation_strategy.trials_as_df)
    ax_client.generation_strategy.trials_as_df.to_csv("./scripts/bayesian_opt/trials.csv")

    print(ax_client.get_best_parameters())
    with open("./scripts/bayesian_opt/best_parameters.pickle", 'wb') as handle:
        pickle.dump(ax_client.get_best_parameters(), handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open("./scripts/bayesian_opt/results.pickle", 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

