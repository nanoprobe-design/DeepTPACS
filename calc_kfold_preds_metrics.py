#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Description: Aggregate k-fold prediction files and compute overall evaluation metrics.
Usage: python calc_kfold_preds_metrics.py <model_name> [desc_tag] [feature_num]
Author: Yibin ZHANG
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


TASK_NAME = "lg(TPACS)"
DEFAULT_DESC_TAG = "gnn"
DEFAULT_FEATURE_NUM = 6
PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class MetricsConfig:
    model_name: str
    desc_tag: str = DEFAULT_DESC_TAG
    feature_num: int = DEFAULT_FEATURE_NUM
    task_name: str = TASK_NAME
    project_root: Path = PROJECT_ROOT

    @property
    def results_dir(self) -> Path:
        return self.project_root / "results" / f"feature_{self.feature_num}" / "training_kfold"

    @property
    def preds_dir(self) -> Path:
        return self.results_dir / "preds"

    @property
    def output_path(self) -> Path:
        return self.results_dir / f"{self.model_name}_{self.task_name}_{self.desc_tag}_all_preds_metrics.csv"


def parse_args(argv: List[str]) -> MetricsConfig:
    if len(argv) < 2:
        raise ValueError("Usage: python calc_kfold_preds_metrics.py <model_name> [desc_tag] [feature_num]")
    desc_tag = argv[2] if len(argv) > 2 else DEFAULT_DESC_TAG
    feature_num = int(argv[3]) if len(argv) > 3 else DEFAULT_FEATURE_NUM
    return MetricsConfig(model_name=argv[1], desc_tag=desc_tag, feature_num=feature_num)


def collect_prediction_files(config: MetricsConfig) -> List[Path]:
    if not config.preds_dir.is_dir():
        raise FileNotFoundError(f"preds dir not found: {config.preds_dir}")

    prefix = f"{config.model_name}_{config.task_name}_{config.desc_tag}"
    pred_files = sorted(
        file_path
        for file_path in config.preds_dir.iterdir()
        if file_path.name.startswith(prefix) and file_path.name.endswith("_preds.csv")
    )
    if not pred_files:
        raise FileNotFoundError(f"no preds files found in: {config.preds_dir}")
    return pred_files


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def load_prediction_frame(pred_file: Path) -> pd.DataFrame:
    frame = pd.read_csv(pred_file)
    required_columns = {"y_true", "y_pred"}
    if not required_columns.issubset(frame.columns):
        raise ValueError(f"missing y_true/y_pred columns in {pred_file}")
    return frame


def summarize_prediction_files(pred_files: Iterable[Path]) -> pd.DataFrame:
    fold_rows = []
    all_true = []
    all_pred = []

    for pred_file in pred_files:
        frame = load_prediction_frame(pred_file)
        y_true = frame["y_true"].to_numpy()
        y_pred = frame["y_pred"].to_numpy()
        all_true.extend(y_true.tolist())
        all_pred.extend(y_pred.tolist())

        row = {"file": pred_file.name}
        row.update(compute_metrics(y_true, y_pred))
        fold_rows.append(row)

    summary_row = {"file": "all_folds"}
    summary_row.update(compute_metrics(np.asarray(all_true), np.asarray(all_pred)))
    return pd.DataFrame(fold_rows + [summary_row])


def main(argv: List[str]) -> None:
    config = parse_args(argv)
    pred_files = collect_prediction_files(config)
    metrics_frame = summarize_prediction_files(pred_files)
    metrics_frame.to_csv(config.output_path, index=False)
    print(f"saved to: {config.output_path}")
    print(metrics_frame.iloc[-1].to_dict())


if __name__ == "__main__":
    main(sys.argv)
