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
from bayesian_utils import ( 
    lr_experiment,
    evaluate_mask_opt, 
    evaluate_norm_loss,
    evaluate_gmm, 
    lyapunov_experiment,
    evaluate_lyapunov,
    dfusn_experiment,
    evaluate_dfusn,
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

task_dict = {'mask_opt': [evaluate_mask_opt, {'data': args.data}, lr_experiment],
                'norm_loss': [evaluate_norm_loss, {'data': args.data}, lr_experiment],
                'lyapunov': [evaluate_lyapunov, {'max_mse': args.max_mse, 'data': args.data}, lyapunov_experiment],
                'gmm': [evaluate_gmm, {'data': args.data}, lr_experiment],
                'dfusn': [evaluate_dfusn, {'data': args.data}, dfusn_experiment]}

if args.task not in task_dict:
    raise ValueError("Undefined task")

evaluate = partial(task_dict[args.task][0], **task_dict[args.task][1])

# evaluate({'dfusn_alpha': 0.92, 'wt_decay': 1e-3})


# %%
# Ref: https://ax.dev/api/_modules/ax/service/ax_client.html#AxClient.create_experiment
ax_client = AxClient()
ax_client.create_experiment(**task_dict[args.task][2])

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

