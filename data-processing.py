#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Description: Prepare descriptor matrices for the classical machine-learning pipeline.
Usage: python data-processing.py
Author: Yibin ZHANG
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parent
ML_ROOT = PROJECT_ROOT / "ml"

SCALER = StandardScaler()
IMPUTER = SimpleImputer(missing_values=np.nan, strategy="mean")


def data_proc(data: pd.DataFrame, data_type: str, save_path_ori: Path) -> None:
    columns = data.columns
    data_x = np.asarray(data[columns[1:]])
    data_y = np.asarray(data[columns[0]])

    if data_type == "descriptors_rdkit":
        data_x[np.isinf(data_x)] = np.nan
        data_x = IMPUTER.fit_transform(data_x)
        data_x = SCALER.fit_transform(data_x)

    data_x_frame = pd.DataFrame(data_x)
    data_y_frame = pd.DataFrame(data_y)
    output_columns = ["y"] + list(range(data_x_frame.shape[1]))
    output = pd.merge(data_y_frame, data_x_frame, left_index=True, right_index=True, sort=False)
    output.columns = output_columns
    output.to_csv(save_path_ori / f"{data_type}_data.csv", index=False)
    print(f"\033[0;32m{data_type} Successfully saved to: {save_path_ori}\033[0m")


def execute(data_path: Path, save_path: Path) -> None:
    save_path_ori = save_path / "ori_data"
    save_path_ori.mkdir(parents=True, exist_ok=True)

    datasets = {
        "morgan": pd.read_csv(data_path / "morgan_fp.csv"),
        "daylight": pd.read_csv(data_path / "daylight_fp.csv"),
        "atompair": pd.read_csv(data_path / "atompair_fp.csv"),
        "toptorsion": pd.read_csv(data_path / "toptorsion_fp.csv"),
        "descriptors_rdkit": pd.read_csv(data_path / "descriptors_rdkit.csv"),
    }

    for name, frame in datasets.items():
        data_proc(frame, name, save_path_ori)


def main() -> None:
    print("begin")
    data_path = ML_ROOT / "input_files"
    execute(data_path, data_path)
    print("end")
    sys.exit(0)


if __name__ == "__main__":
    main()
