# Expected demo output

The reference demo was validated with the released checkpoint on the tested environment recorded in
`environment/TESTED_ENVIRONMENT.md`.

Run:

```bash
python predict_csv.py --input demo/input_smiles.csv --output demo/demo_output.csv --device auto
```

A successful run creates `demo/demo_output.csv` with the same input rows plus:

- `predicted_log10_TPACS`: DeepTPACS output for the manuscript target `lg(TPACS)`;
- `predicted_TPACS`: `10 ** predicted_log10_TPACS`, included for convenience.

The validated run used `cuda:0`, predicted three molecules, and completed in **5.297 s**. The exact
reference CSV is committed as `demo/reference_output.csv`.

| Molecule | predicted_log10_TPACS | predicted_TPACS |
|---|---:|---:|
| TPP-2MP | 1.38685810566 | 24.3701445767 |
| 1,4-BPA | 2.57553172112 | 376.297836271 |
| TPE-TAC | 3.28325915337 | 1919.81399548 |

Small floating-point differences can occur across hardware/software builds, but the released checkpoint,
input SMILES, and tested environment are recorded so that the deposited reference can be checked directly.
