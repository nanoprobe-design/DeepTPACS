#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Description: Run hyperparameter optimization for classical machine-learning models.
Usage: python model_params_opt.py <model_name> [--descriptors daylight] [--n_trials 300]
Author: Yibin ZHANG
"""

import argparse
import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from optuna.samplers import TPESampler
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor


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


@dataclass(frozen=True)
class OptConfig:
    model_name: str
    full_path: Path
    descriptors: str = "daylight"
    n_trials: int = 300
    random_seed: int = RANDOM_SEED

    @property
    def data_dir(self) -> Path:
        return self.full_path / "input_files" / "ori_data"

    @property
    def checkpoint_dir(self) -> Path:
        return self.full_path / "checkpoint"

    @property
    def results_dir(self) -> Path:
        return self.full_path / "results"

    @property
    def data_file(self) -> Path:
        return self.data_dir / f"{self.descriptors}_data.csv"

    @property
    def output_params_file(self) -> Path:
        return self.results_dir / f"{self.model_name}_best_params.txt"


def build_trial_params(trial: optuna.Trial, model_name: str) -> Dict:
    if model_name == "KNN":
        return {
            "n_neighbors": trial.suggest_categorical("n_neighbors", [1, 2, 3, 4, 5]),
            "weights": trial.suggest_categorical("weights", ["distance", "uniform"]),
            "metric": trial.suggest_categorical("metric", ["euclidean", "minkowski"]),
            "p": trial.suggest_categorical("p", [1, 2]),
        }
    if model_name == "XGB":
        return {
            "tree_method": trial.suggest_categorical("tree_method", ["gpu_hist"]),
            "objective": trial.suggest_categorical("objective", ["reg:squarederror"]),
            "max_depth": trial.suggest_categorical("max_depth", [5, 10, 15, 20, 25]),
            "learning_rate": trial.suggest_categorical("learning_rate", [0.01, 0.02, 0.05, 0.1, 0.15]),
            "n_estimators": trial.suggest_categorical("n_estimators", [10, 100, 200, 500, 1000]),
            "min_child_weight": trial.suggest_categorical("min_child_weight", [0, 2, 5, 10, 20]),
            "max_delta_step": trial.suggest_categorical("max_delta_step", [0, 0.2, 0.6, 1, 2]),
            "subsample": trial.suggest_categorical("subsample", [0.6, 0.7, 0.8, 0.85, 0.95]),
            "colsample_bytree": trial.suggest_categorical("colsample_bytree", [0.5, 0.6, 0.7, 0.8, 0.9]),
            "reg_alpha": trial.suggest_categorical("reg_alpha", [0, 0.25, 0.5, 0.75, 1]),
            "reg_lambda": trial.suggest_categorical("reg_lambda", [0.2, 0.4, 0.6, 0.8, 1]),
            "scale_pos_weight": trial.suggest_categorical("scale_pos_weight", [0.2, 0.4, 0.6, 0.8, 1]),
        }
    if model_name == "RF":
        return {
            "n_estimators": trial.suggest_categorical("n_estimators", [10, 50, 100, 200, 500]),
            "max_depth": trial.suggest_categorical("max_depth", [5, 10, 50, 100, 200]),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "min_samples_split": trial.suggest_categorical("min_samples_split", list(range(2, 7))),
        }
    if model_name == "GBDT":
        return {
            "loss": trial.suggest_categorical("loss", ["squared_error"]),
            "learning_rate": trial.suggest_categorical("learning_rate", [0.01, 0.05, 0.1, 1]),
            "min_samples_split": trial.suggest_categorical("min_samples_split", list(np.linspace(0.1, 0.5, 12))),
            "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", list(np.linspace(0.1, 0.5, 12))),
            "max_depth": trial.suggest_categorical("max_depth", [3, 5, 7]),
            "max_features": trial.suggest_categorical("max_features", ["sqrt"]),
            "criterion": trial.suggest_categorical("criterion", ["friedman_mse"]),
            "subsample": trial.suggest_categorical("subsample", [0.5, 0.75, 0.95]),
            "n_estimators": trial.suggest_categorical("n_estimators", [10, 50, 100, 200, 500]),
        }
    if model_name == "ADA":
        return {
            "n_estimators": trial.suggest_categorical("n_estimators", [10, 50, 100, 200, 500]),
            "learning_rate": trial.suggest_categorical("learning_rate", [0.01, 0.05, 0.1, 1]),
            "loss": trial.suggest_categorical("loss", ["square", "exponential"]),
        }
    raise ValueError(f"Unsupported model_name: {model_name}")


def build_model(model_name: str, params: Dict):
    if model_name == "RF":
        return RandomForestRegressor(**params)
    if model_name == "GBDT":
        return GradientBoostingRegressor(**params)
    if model_name == "ADA":
        return AdaBoostRegressor(**params)
    if model_name == "KNN":
        return KNeighborsRegressor(**params)
    raise ValueError(f"Unsupported sklearn model: {model_name}")


def train_fold(config: OptConfig, fold: int, train_x, train_y, test_x, test_y, params: Dict):
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if config.model_name == "XGB":
        model_file = config.checkpoint_dir / f"{config.model_name}_fold_{fold}.json"
        d_train = xgb.DMatrix(train_x, train_y)
        d_test = xgb.DMatrix(test_x, test_y)
        model = xgb.train(params, d_train, 1000, [(d_train, "train"), (d_test, "valid")], early_stopping_rounds=100, verbose_eval=100)
        model.save_model(model_file)
        return model.predict(d_train), train_y, model.predict(d_test), test_y

    model_file = config.checkpoint_dir / f"{config.model_name}_fold_{fold}.pkl"
    model = build_model(config.model_name, params)
    model.fit(train_x, train_y)
    with model_file.open("wb") as file:
        pickle.dump(model, file)
    return model.predict(train_x), train_y, model.predict(test_x), test_y


def compute_metric_summary(y_true, y_pred):
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "pearson": np.corrcoef(y_pred, y_true)[0, 1],
    }


def load_dataset(config: OptConfig) -> np.ndarray:
    return np.asarray(pd.read_csv(config.data_file))


def make_objective(config: OptConfig, np_data: np.ndarray):
    def objective(trial: optuna.Trial):
        params = build_trial_params(trial, config.model_name)
        kf = KFold(n_splits=10, shuffle=True, random_state=config.random_seed)
        indices = np.arange(len(np_data))
        train_outputs_all, train_labels_all = [], []
        test_outputs_all, test_labels_all = [], []

        for fold, (train_index, test_index) in enumerate(kf.split(indices)):
            train_data = np_data[train_index, :]
            test_data = np_data[test_index, :]
            train_y, train_x = train_data[:, 0], train_data[:, 1:]
            test_y, test_x = test_data[:, 0], test_data[:, 1:]
            train_outputs, train_labels, test_outputs, test_labels = train_fold(
                config, fold, train_x, train_y, test_x, test_y, params
            )
            train_outputs_all.extend(train_outputs.tolist())
            train_labels_all.extend(train_labels.tolist())
            test_outputs_all.extend(test_outputs.tolist())
            test_labels_all.extend(test_labels.tolist())

            train_metrics = compute_metric_summary(train_labels, train_outputs)
            test_metrics = compute_metric_summary(test_labels, test_outputs)
            print(
                "Fold:{}, MSE:{:.4f}/{:.4f}, MAE:{:.4f}/{:.4f}, R2:{:.4f}/{:.4f}, Pearson:{:.4f}/{:.4f}".format(
                    fold,
                    train_metrics["mse"], test_metrics["mse"],
                    train_metrics["mae"], test_metrics["mae"],
                    train_metrics["r2"], test_metrics["r2"],
                    train_metrics["pearson"], test_metrics["pearson"],
                )
            )

        train_metrics = compute_metric_summary(np.asarray(train_labels_all), np.asarray(train_outputs_all))
        test_metrics = compute_metric_summary(np.asarray(test_labels_all), np.asarray(test_outputs_all))
        print(
            "MSE:{:.4f}/{:.4f}, MAE:{:.4f}/{:.4f}, R2:{:.4f}/{:.4f}, Pearson:{:.4f}/{:.4f}".format(
                train_metrics["mse"], test_metrics["mse"],
                train_metrics["mae"], test_metrics["mae"],
                train_metrics["r2"], test_metrics["r2"],
                train_metrics["pearson"], test_metrics["pearson"],
            )
        )
        return test_metrics["mse"]

    return objective


def parse_args() -> OptConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="GBDT, XGB, RF, KNN, ADA")
    parser.add_argument("--full_path", default=str(ML_ROOT))
    parser.add_argument("--descriptors", default="daylight")
    parser.add_argument("--n_trials", type=int, default=300)
    args = parser.parse_args()
    return OptConfig(
        model_name=args.model_name,
        full_path=Path(args.full_path),
        descriptors=args.descriptors,
        n_trials=args.n_trials,
    )


def main() -> None:
    configure_tensorflow()
    set_seed(RANDOM_SEED)
    config = parse_args()
    np_data = load_dataset(config)
    study = optuna.create_study(sampler=TPESampler(), direction="minimize")
    study.optimize(make_objective(config, np_data), n_trials=config.n_trials, n_jobs=1)
    trial = study.best_trial
    config.results_dir.mkdir(parents=True, exist_ok=True)
    with config.output_params_file.open("w", encoding="utf-8") as file:
        for key, value in trial.params.items():
            file.write(f"{key} : {value}\n")
    print(f"{config.model_name} best params is {trial.params}")
    print(f"{config.model_name} best params is saved to {config.output_params_file}")


if __name__ == "__main__":
    main()
