#!/usr/bin/env python3
"""Validate the released DeepTPACS checkpoint, parameter file, and key tensor shapes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "config" / "model_release.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> int:
    meta = json.loads(META.read_text(encoding="utf-8"))
    ckpt = ROOT / meta["checkpoint"]
    params = ROOT / meta["best_params_index_file"]
    failures = []

    if sha256(ckpt) != meta["checkpoint_sha256"]:
        failures.append("checkpoint SHA-256 mismatch")
    if sha256(params) != meta["best_params_sha256"]:
        failures.append("parameter-file SHA-256 mismatch")

    obj = load_checkpoint(ckpt)
    if not isinstance(obj, dict) or "model_state_dict" not in obj:
        failures.append("checkpoint does not contain model_state_dict")
    else:
        state = obj["model_state_dict"]
        checks = {
            "gnn.init_context.project_node.0.weight": (300, 30),
            "gnn.init_context.project_edge1.0.weight": (300, 43),
            "predict.1.weight": (1, 300),
        }
        for key, expected_shape in checks.items():
            actual = tuple(state[key].shape) if key in state else None
            if actual != expected_shape:
                failures.append(f"{key}: expected {expected_shape}, got {actual}")
        if len(state) != meta["checkpoint_tensor_count"]:
            failures.append(f"expected {meta['checkpoint_tensor_count']} tensors, got {len(state)}")
        if obj.get("timestep") != meta["checkpoint_timestep"]:
            failures.append(f"expected checkpoint timestep {meta['checkpoint_timestep']}, got {obj.get('timestep')}")

    if failures:
        print("DeepTPACS release verification: FAILED")
        for item in failures:
            print(" -", item)
        return 1

    print("DeepTPACS release verification: PASS")
    print(" checkpoint:", meta["checkpoint"])
    print(" SHA-256:", meta["checkpoint_sha256"])
    print(" graph feature size:", meta["graph_feature_size"])
    print(" GNN layers:", meta["num_layers"])
    print(" readout timesteps:", meta["num_timesteps"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
