#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate RDKit descriptors and fingerprints for the classical-ML pipeline.

By default, the script uses the public 100-row sample dataset:
    data_process/TPACS_sample_data.csv

Examples
--------
python get-descriptors.py
python get-descriptors.py --input data_process/TPACS_sample_data.csv --label "log(TPACS)"
"""

import argparse
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from rdkit import DataStructs
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parent
ML_ROOT = PROJECT_ROOT / "ml"


def save_csv(data: pd.DataFrame, name: str, save_path: Path) -> None:
    save_path.mkdir(parents=True, exist_ok=True)
    output_path = save_path / name
    data.to_csv(output_path, index=False)
    print(f"Successfully saved to: {output_path}")


def resolve_label(data: pd.DataFrame, requested: Optional[str]) -> str:
    if requested:
        if requested not in data.columns:
            raise KeyError(f"Requested label column '{requested}' not found. Available columns: {list(data.columns)}")
        return requested
    for candidate in ("log(TPACS)", "lg(TPACS)"):
        if candidate in data.columns:
            return candidate
    raise KeyError("Could not find a TPACS target column. Expected 'log(TPACS)' or 'lg(TPACS)'.")


def get_fp(data_path: Path, save_path: Path, label_name: Optional[str] = None) -> None:
    data = pd.read_csv(data_path)
    if "SMILES" not in data.columns:
        raise KeyError("Input CSV must contain a 'SMILES' column.")

    label_name = resolve_label(data, label_name)
    data["mol"] = [Chem.MolFromSmiles(smiles) for smiles in data["SMILES"]]

    invalid_mask = data["mol"].isna()
    if invalid_mask.any():
        invalid_smiles = data.loc[invalid_mask, "SMILES"].tolist()
        raise ValueError(f"Invalid SMILES found: {invalid_smiles[:5]}")

    y = data[label_name]

    descriptors_all = [descriptor[0] for descriptor in Descriptors._descList]
    descriptor_calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptors_all)
    descriptors = pd.DataFrame([descriptor_calc.CalcDescriptors(mol) for mol in data["mol"]])
    descriptors = pd.merge(y, descriptors, left_index=True, right_index=True, sort=False)
    descriptors.columns = ["y"] + descriptors_all
    save_csv(descriptors, "descriptors_rdkit.csv", save_path)

    fingerprint_builders = {
        "morgan_fp.csv": lambda mol: Chem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048),
        "daylight_fp.csv": Chem.RDKFingerprint,
        "atompair_fp.csv": Chem.GetHashedAtomPairFingerprintAsBitVect,
        "toptorsion_fp.csv": Chem.GetHashedTopologicalTorsionFingerprintAsBitVect,
    }

    for filename, builder in fingerprint_builders.items():
        fingerprints = [builder(mol) for mol in data["mol"]]
        arrays = []
        for fingerprint in fingerprints:
            array = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fingerprint, array)
            arrays.append(array)
        fp_frame = pd.DataFrame(arrays)
        output = pd.merge(y, fp_frame, left_index=True, right_index=True, sort=False)
        output.columns = ["y"] + list(range(fp_frame.shape[1]))
        save_csv(output, filename, save_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "data_process" / "TPACS_sample_data.csv"),
        help="Raw CSV containing SMILES and the TPACS target column.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Target column name. If omitted, 'log(TPACS)' then 'lg(TPACS)' are detected automatically.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ML_ROOT / "input_files"),
        help="Directory for generated descriptor/fingerprint CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    get_fp(Path(args.input), Path(args.output_dir), args.label)


if __name__ == "__main__":
    main()
