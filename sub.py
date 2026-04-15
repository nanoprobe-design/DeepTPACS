#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Description: Launch batch metric aggregation jobs for multiple GNN models.
Usage: python sub.py
Author: Yibin ZHANG
"""

import subprocess


MODELS = ["DeepTPACS", "gat", "gcn", "mpnn", "weave"]


for model_name in MODELS:
    subprocess.Popen(["python", "calc_kfold_preds_metrics.py", model_name])
