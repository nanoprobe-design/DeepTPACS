#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Description: Train or evaluate classical machine-learning models and save results.
Usage: python ml_train.py <model_name> [--descriptors daylight] [--mode train]
Author: Yibin ZHANG
"""

import argparse
import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR


PROJECT_ROOT = Path(__file__).resolve().parent
ML_ROOT = PROJECT_ROOT / "ml"
RANDOM_SEED = 42


def configure_tensorflow() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as exc:
        print(exc)


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


class XgbR2Printer(xgb.callback.TrainingCallback):
    def __init__(self, dtrain, dvalid, period: int = 100):
        self.dtrain = dtrain
        self.dvalid = dvalid
        self.period = period

    def after_iteration(self, model, epoch, evals_log):
        if epoch % self.period != 0:
            return False
        train_pred = model.predict(self.dtrain)
        valid_pred = model.predict(self.dvalid)
        train_r2 = r2_score(self.dtrain.get_label(), train_pred)
        valid_r2 = r2_score(self.dvalid.get_label(), valid_pred)
        print(f"[{epoch}] train-r2:{train_r2:.5f} valid-r2:{valid_r2:.5f}")
        return False


@dataclass(frozen=True)
class TrainConfig:
    model_name: str
    full_path: Path
    descriptors: str
    mode: str
    random_seed: int = RANDOM_SEED

    @property
    def data_dir(self) -> Path:
        return self.full_path / "input_files" / "ori_data"

    @property
    def results_dir(self) -> Path:
        return self.full_path / "results"

    @property
    def checkpoint_dir(self) -> Path:
        return self.full_path / "checkpoint"

    @property
    def data_file(self) -> Path:
        return self.data_dir / f"{self.descriptors}_data.csv"

    @property
    def params_file(self) -> Path:
        return self.results_dir / f"{self.model_name}_best_params.txt"


def plot_predictions(y_true, y_pred) -> None:
    fig = plt.figure()
    fig.subplots()
    plt.scatter(y_true, y_pred, 10, label="prediction")
    plt.xlabel("true")
    plt.ylabel("predict")
    plt.legend(loc="best")
    plt.grid()
    plt.show()


def default_model_params() -> Dict[str, Dict]:
    return {
        "RF": {
            "n_estimators": 200,
            "max_depth": 100,
            "max_features": "sqrt",
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "n_jobs": -1,
            "random_state": RANDOM_SEED,
        },
        "GBDT": {
            "loss": "squared_error",
            "learning_rate": 0.1,
            "n_estimators": 1000,
            "max_depth": 15,
            "max_features": "sqrt",
            "random_state": RANDOM_SEED,
        },
        "ADA": {
            "n_estimators": 50,
            "learning_rate": 0.1,
            "loss": "square",
        },
        "KNN": {
            "n_neighbors": 3,
            "weights": "distance",
            "metric": "minkowski",
            "p": 1,
        },
        "SVM": {
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale",
            "epsilon": 0.1,
        },
        "XGB": {
            "tree_method": "gpu_hist",
            "objective": "reg:squarederror",
            "eval_metric": "mae",
            "max_depth": 8,
            "min_child_weight": 2,
            "gamma": 0,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
            "alpha": 0,
            "lambda": 0.6,
            "eta": 0.01,
            "seed": RANDOM_SEED,
        },
    }


def build_model(model_name: str, params: Dict):
    if model_name == "RF":
        return RandomForestRegressor(**params)
    if model_name == "GBDT":
        return GradientBoostingRegressor(**params)
    if model_name == "ADA":
        return AdaBoostRegressor(**params)
    if model_name == "KNN":
        return KNeighborsRegressor(**params)
    if model_name == "SVM":
        return SVR(**params)
    raise ValueError(f"Unsupported sklearn model: {model_name}")


def load_best_params(config: TrainConfig) -> Dict:
    params: Dict[str, object] = {}
    with config.params_file.open("r", encoding="utf-8") as file:
        for raw_line in file:
            key, value = raw_line.split(":", 1)
            value = value.strip()
            try:
                parsed = float(value)
                if parsed.is_integer():
                    parsed = int(parsed)
                params[key.strip()] = parsed
            except ValueError:
                params[key.strip()] = value
    if config.model_name == "XGB":
        params["objective"] = "reg:squarederror"
    return params


def load_dataset(config: TrainConfig) -> np.ndarray:
    data_frame = pd.read_csv(config.data_file)
    return np.asarray(data_frame)


def checkpoint_path(config: TrainConfig, fold: int, suffix: str) -> Path:
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return config.checkpoint_dir / f"{config.model_name}_fold_{fold}.{suffix}"


def train_or_predict_fold(config: TrainConfig, model_name: str, fold: int, train_x, train_y, test_x, test_y, params: Dict):
    if model_name == "XGB":
        model_file = checkpoint_path(config, fold, "json")
        d_train = xgb.DMatrix(train_x, train_y)
        d_test = xgb.DMatrix(test_x, test_y)
        if config.mode == "train":
            watchlist = [(d_train, "train"), (d_test, "valid")]
            model = xgb.train(
                params,
                d_train,
                500,
                watchlist,
                early_stopping_rounds=100,
                verbose_eval=100,
                callbacks=[XgbR2Printer(d_train, d_test, period=100)],
            )
            model.save_model(model_file)
        else:
            model = xgb.Booster(model_file=str(model_file))
        return model.predict(d_train), train_y, model.predict(d_test), test_y

    model_file = checkpoint_path(config, fold, "pkl")
    if config.mode == "train":
        model = build_model(model_name, params)
        model.fit(train_x, train_y)
        with model_file.open("wb") as file:
            pickle.dump(model, file)
    else:
        with model_file.open("rb") as file:
            model = pickle.load(file)
    return model.predict(train_x), train_y, model.predict(test_x), test_y


def compute_fold_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "pearson": np.corrcoef(y_pred, y_true)[0, 1],
    }


def run_kfold(config: TrainConfig, np_data: np.ndarray, params: Dict):
    if len(np_data) == 0:
        raise ValueError("Input data is empty; check the input file path and contents.")

    kf = KFold(n_splits=10, shuffle=True, random_state=config.random_seed)
    indices = np.arange(len(np_data))
    train_outputs_all, train_labels_all = [], []
    test_outputs_all, test_labels_all = [], []
    fold_metrics = []

    for fold, (train_index, test_index) in enumerate(kf.split(indices)):
        train_data = np_data[train_index, :]
        test_data = np_data[test_index, :]
        train_y, train_x = train_data[:, 0], train_data[:, 1:]
        test_y, test_x = test_data[:, 0], test_data[:, 1:]

        train_outputs, train_labels, test_outputs, test_labels = train_or_predict_fold(
            config,
            config.model_name,
            fold,
            train_x,
            train_y,
            test_x,
            test_y,
            params,
        )
        train_outputs_all.extend(train_outputs.tolist())
        train_labels_all.extend(train_labels.tolist())
        test_outputs_all.extend(test_outputs.tolist())
        test_labels_all.extend(test_labels.tolist())

        train_metrics = compute_fold_metrics(train_labels, train_outputs)
        test_metrics = compute_fold_metrics(test_labels, test_outputs)
        fold_metrics.append(
            {
                "fold": fold,
                "train_mse": train_metrics["mse"],
                "test_mse": test_metrics["mse"],
                "train_mae": train_metrics["mae"],
                "test_mae": test_metrics["mae"],
                "train_r2": train_metrics["r2"],
                "test_r2": test_metrics["r2"],
                "train_pearson": train_metrics["pearson"],
                "test_pearson": test_metrics["pearson"],
            }
        )
        print(
            "Fold:{}, MSE:{:.4f}/{:.4f}, MAE:{:.4f}/{:.4f}, R2:{:.4f}/{:.4f}, Pearson:{:.4f}/{:.4f}".format(
                fold,
                train_metrics["mse"], test_metrics["mse"],
                train_metrics["mae"], test_metrics["mae"],
                train_metrics["r2"], test_metrics["r2"],
                train_metrics["pearson"], test_metrics["pearson"],
            )
        )

    return train_outputs_all, train_labels_all, test_outputs_all, test_labels_all, fold_metrics


def save_results(config: TrainConfig, train_outputs_all, train_labels_all, test_outputs_all, test_labels_all, fold_metrics) -> None:
    config.results_dir.mkdir(parents=True, exist_ok=True)
    train_metrics = compute_fold_metrics(np.asarray(train_labels_all), np.asarray(train_outputs_all))
    test_metrics = compute_fold_metrics(np.asarray(test_labels_all), np.asarray(test_outputs_all))
    print(
        "MSE:{:.4f}/{:.4f}, MAE:{:.4f}/{:.4f}, R2:{:.4f}/{:.4f}, Pearson:{:.4f}/{:.4f}".format(
            train_metrics["mse"], test_metrics["mse"],
            train_metrics["mae"], test_metrics["mae"],
            train_metrics["r2"], test_metrics["r2"],
            train_metrics["pearson"], test_metrics["pearson"],
        )
    )

    summary_frame = pd.DataFrame(
        {
            "type": ["mse", "mae", "r2", "pearson"],
            "train": [train_metrics["mse"], train_metrics["mae"], train_metrics["r2"], train_metrics["pearson"]],
            "test": [test_metrics["mse"], test_metrics["mae"], test_metrics["r2"], test_metrics["pearson"]],
        }
    )
    summary_frame.to_csv(config.results_dir / f"{config.model_name}_results.csv", index=False)
    pd.DataFrame(fold_metrics).to_csv(config.results_dir / f"{config.model_name}_fold_metrics.csv", index=False)
    pd.DataFrame({"predictions": test_outputs_all, "labels": test_labels_all}).to_csv(
        config.results_dir / f"{config.model_name}_predictions.csv",
        index=False,
    )
    plot_predictions(test_labels_all, test_outputs_all)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="SVM, XGB, RF, KNN, ADA")
    parser.add_argument("--full_path", default=str(ML_ROOT))
    parser.add_argument("--descriptors", default="daylight")
    parser.add_argument("--mode", default="train")
    args = parser.parse_args()
    return TrainConfig(
        model_name=args.model_name,
        full_path=Path(args.full_path),
        descriptors=args.descriptors,
        mode=args.mode,
    )


def main() -> None:
    configure_tensorflow()
    set_seed(RANDOM_SEED)
    config = parse_args()
    params = load_best_params(config)
    print(params)
    np_data = load_dataset(config)
    outputs = run_kfold(config, np_data, params)
    save_results(config, *outputs)


if __name__ == "__main__":
    main()
