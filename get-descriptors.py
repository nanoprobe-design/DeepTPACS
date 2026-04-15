#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Description: Generate RDKit descriptors and fingerprints for classical ML inputs.
Usage: python get-descriptors.py
Author: Yibin ZHANG
"""

import sys
import warnings
from pathlib import Path

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
    output_path = save_path / name
    data.to_csv(output_path, index=False)
    print(f"\033[0;32mSuccessfully saved to: {output_path}\033[0m")


def get_fp(data_path: Path, save_path: Path, label_name: str) -> None:
    data = pd.read_csv(data_path)
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
        output.columns = ["y"] + list(range(2048))
        save_csv(output, filename, save_path)


def main() -> None:
    data_path = PROJECT_ROOT / "data_process" / "TPCS_attentivefp_ori_augment.csv"
    save_path = ML_ROOT / "input_files"
    save_path.mkdir(parents=True, exist_ok=True)
    get_fp(data_path, save_path, "lg(TPACS)")
    print("end")
    sys.exit(0)


if __name__ == "__main__":
    main()
