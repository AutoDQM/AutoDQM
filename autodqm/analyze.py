#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib as mpl
import compare_hists
import pandas as pd
import json
import argparse
import sys


parser = argparse.ArgumentParser()
parser.add_argument(
    "args",
    type=str,
    help="first one json file, second is subsystem",
    nargs='+'
)

args = parser.parse_args()

config_dir = "../config"
subsystem = args.args[1]
data_series = "Run2018"
data_sample = "SingleMuon"
ref_series = "Run2018"
ref_sample = "SingleMuon"
chunk_index = 0
chunk_size = 9999
dqmSource = "Offline"

plotdir = "plots"

datadict = json.load(open(args.args[0]))

## create the csv files for storing the scores
from pathlib import Path

Path("csv").mkdir(parents=True, exist_ok=True)
with open("csv/beta_binomial.csv", "w+") as myfile:
    myfile.write("histname,bb_pull,bb_chi2,ref_run,data_run\n")
with open("csv/ks.csv", "w+") as myfile:
    myfile.write("histname,ks,ref_run,data_run\n")
with open("csv/pullvals.csv", "w+") as myfile:
    myfile.write("histname,maxpull,chi2,ref_run,data_run\n")

for data_path in datadict:
    runnum_idx = data_path.find("_R000") + 5  # data_path[-11:-5]
    data_run = data_path[runnum_idx : runnum_idx + 6]
    ref_path = datadict[data_path]
    runnum_idx = ref_path.find("_R000") + 5
    ref_run = ref_path[runnum_idx : runnum_idx + 6]
    compare_hists.process(
        chunk_index,
        chunk_size,
        config_dir,
        dqmSource,
        subsystem,
        data_series,
        data_sample,
        data_run,
        data_path,
        ref_series,
        ref_sample,
        ref_run,
        ref_path,
        output_dir="./out/",
        plugin_dir="../plugins/",
    )

bb = pd.read_csv("csv/beta_binomial.csv")
pv = pd.read_csv("csv/pullvals.csv")
ks = pd.read_csv("csv/ks.csv")

merged = bb.merge(pv, how="left", on=["histname", "data_run", "ref_run"])
merged = merged.merge(ks, how="left", on=["histname", "data_run", "ref_run"])

merged1d = merged[~np.isnan(merged.ks)]
merged2d = merged[~np.isnan(merged.maxpull)]


## maybe using numpy.hist would work better which allow bar
Path("plots").mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots()
xmax = 37
histkwarg = {"bins": 20, "alpha": 0.5, "range": (0, xmax)}
ax.hist(
    np.clip(merged2d.maxpull, a_min=None, a_max=xmax), label="max_pull", **histkwarg
)
_, bins, _ = ax.hist(merged2d.bb_pull, label="beta_binomial", **histkwarg)
xlabels = ax.get_xticks().tolist()
xlabels = [str(x) for x in xlabels[0:-1]]
xlabels[-1] += "+"
ax.set_xticklabels(xlabels)
ax.set_title(f"{subsystem} pull values and beta-binomial")
ax.legend()
fig.savefig(f"plots/pullvals_{subsystem}.png", bbox_inches="tight")

fig2, ax2 = plt.subplots()
xmax = 105
histkwarg = {"bins": 21, "alpha": 0.5, "range": (0, xmax)}
ax2.hist(np.clip(merged2d.chi2, a_min=None, a_max=xmax), label="chi2", **histkwarg)
_, bins, _ = ax2.hist(
    np.clip(merged2d.bb_chi2, a_min=None, a_max=xmax),
    label="beta_binomial chi2",
    **histkwarg,
)
xlabels = ax2.get_xticks().tolist()
xlabels = [str(x) for x in xlabels[0:-1]]
xlabels[-1] += "+"
ax2.set_xticklabels(xlabels)
ax2.set_title(f"{subsystem} chi2 and beta-binomial chi2")
ax2.legend()
fig2.savefig(f"plots/chi2_{subsystem}.png", bbox_inches="tight")
