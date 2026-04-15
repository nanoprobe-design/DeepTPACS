#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Description: Launch batch runs for the classical machine-learning training scripts.
Usage: python sub_ml.py
Author: Yibin ZHANG
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def run_script(script_name: str, *args: str) -> None:
    subprocess.run([sys.executable, str(PROJECT_ROOT / script_name), *args], check=True)


def main() -> None:
    print("begin")
    # Hyperparameter optimization
    # run_script("model_params_opt.py", "GBDT")
    # run_script("model_params_opt.py", "RF")
    # run_script("model_params_opt.py", "KNN")
    # run_script("model_params_opt.py", "ADA")
    # run_script("model_params_opt.py", "XGB")

    # Model training
    run_script("ml_train.py", "XGB")
    run_script("ml_train.py", "GBDT")
    run_script("ml_train.py", "RF")
    run_script("ml_train.py", "KNN")
    run_script("ml_train.py", "ADA")
    print("Done")


if __name__ == "__main__":
    main()
