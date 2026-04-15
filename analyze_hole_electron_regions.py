#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Description: Analyze hole/electron cube files over user-defined atom regions.
Usage: python analyze_hole_electron_regions.py --hole <hole.cub> --electron <electron.cub> --red <atoms> --blue <atoms> [--cutoff 2.5]
Author: Yibin ZHANG

Analyze hole/electron density distribution over two user-defined regions
from Gaussian/standard cube files.

Method:
1. Read hole.cub and electron.cub
2. Extract atomic coordinates and volumetric grids
3. Define two regions (red and blue) by atom indices
4. Assign each grid point to red or blue according to nearest-atom-group rule
5. Optionally ignore grid points too far from all selected atoms using a distance cutoff
6. Integrate densities on each region

Outputs:
- Ph(R), Ph(B)
- Pe(R), Pe(B)
- S_target   = Ph(R) * Pe(B)
- S_mismatch = Ph(B) * Pe(R)

Author: Yibin ZHANG
Red atoms      : 1,3,4,5,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,53,54,55,56,57,58,59,60,61,62,63,64,65,66,69,70,77,78,79,80,81,82,83,87,88,100,101,102,103,104,105,106,107,108,109,110
Blue atoms     : 2,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,42,43,44,45,46,47,48,49,50,51,52,67,68,71,72,73,74,75,76,84,85,86,89,90,91,92,93,94,95,96,97,98,99
Instruction:
python analyze_hole_electron_regions.py
  --hole Fig2_1_hole.cub
  --electron Fig2_1_electron.cub
  --red 1,3,4,5,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,53,54,55,56,57,58,59,60,61,62,63,64,65,66,69,70,77,78,79,80,81,82,83,87,88,100,101,102,103,104,105,106,107,108,109,110 \
  --blue 2,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,42,43,44,45,46,47,48,49,50,51,52,67,68,71,72,73,74,75,76,84,85,86,89,90,91,92,93,94,95,96,97,98,99
  --cutoff 2.5
"""


import argparse
import sys
from typing import List, Tuple

import numpy as np


BOHR_TO_ANG = 0.529177210903


def parse_atom_range(text: str) -> List[int]:
    """
    Parse strings like:
        "1-5,8,10-12"
    into:
        [1,2,3,4,5,8,10,11,12]
    Atom indices are kept 1-based here.
    """
    items = text.replace(" ", "").split(",")
    result = []
    for item in items:
        if not item:
            continue
        if "-" in item:
            a, b = item.split("-")
            a = int(a)
            b = int(b)
            if a > b:
                raise ValueError(f"Invalid range: {item}")
            result.extend(range(a, b + 1))
        else:
            result.append(int(item))
    # remove duplicates while preserving order
    seen = set()
    out = []
    for x in result:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def read_cube(filename: str):
    """
    Read a standard cube file.

    Returns a dict with:
        natoms
        origin_bohr: (3,)
        dims: (nx, ny, nz)
        axes_bohr: (3,3)  # axis vectors for x/y/z grid increments
        atom_numbers: (natoms,)
        atom_charges: (natoms,)
        atom_coords_bohr: (natoms,3)
        data: (nx,ny,nz)
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    if len(lines) < 6:
        raise ValueError(f"{filename} does not look like a valid cube file.")

    # First two lines are comments
    header2 = lines[2].split()
    if len(header2) < 4:
        raise ValueError(f"Failed to parse line 3 in {filename}")

    natoms = int(header2[0])
    origin_bohr = np.array([float(header2[1]), float(header2[2]), float(header2[3])], dtype=float)

    dim_lines = lines[3:6]
    dims = []
    axes_bohr = []
    for line in dim_lines:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Failed to parse grid line in {filename}: {line}")
        n = int(parts[0])
        vec = [float(parts[1]), float(parts[2]), float(parts[3])]
        dims.append(abs(n))
        axes_bohr.append(vec)

    dims = tuple(dims)
    axes_bohr = np.array(axes_bohr, dtype=float)

    atom_start = 6
    atom_end = atom_start + abs(natoms)
    atom_lines = lines[atom_start:atom_end]

    atom_numbers = []
    atom_charges = []
    atom_coords_bohr = []
    for line in atom_lines:
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"Failed to parse atom line in {filename}: {line}")
        atom_numbers.append(int(float(parts[0])))
        atom_charges.append(float(parts[1]))
        atom_coords_bohr.append([float(parts[2]), float(parts[3]), float(parts[4])])

    atom_numbers = np.array(atom_numbers, dtype=int)
    atom_charges = np.array(atom_charges, dtype=float)
    atom_coords_bohr = np.array(atom_coords_bohr, dtype=float)

    # Remaining numbers are volumetric data
    data_tokens = []
    for line in lines[atom_end:]:
        data_tokens.extend(line.split())

    expected = dims[0] * dims[1] * dims[2]
    if len(data_tokens) < expected:
        raise ValueError(
            f"{filename}: not enough volumetric data. "
            f"Expected {expected}, got {len(data_tokens)}"
        )

    data = np.array([float(x) for x in data_tokens[:expected]], dtype=float)
    data = data.reshape(dims)

    return {
        "natoms": abs(natoms),
        "origin_bohr": origin_bohr,
        "dims": dims,
        "axes_bohr": axes_bohr,
        "atom_numbers": atom_numbers,
        "atom_charges": atom_charges,
        "atom_coords_bohr": atom_coords_bohr,
        "data": data,
    }


def check_compatibility(cube1, cube2):
    """
    Ensure hole and electron cube files share the same grid and geometry.
    """
    if cube1["natoms"] != cube2["natoms"]:
        raise ValueError("hole.cub and electron.cub have different numbers of atoms")

    if cube1["dims"] != cube2["dims"]:
        raise ValueError("hole.cub and electron.cub have different grid dimensions")

    if not np.allclose(cube1["origin_bohr"], cube2["origin_bohr"], atol=1e-8):
        raise ValueError("hole.cub and electron.cub have different origins")

    if not np.allclose(cube1["axes_bohr"], cube2["axes_bohr"], atol=1e-8):
        raise ValueError("hole.cub and electron.cub have different axis vectors")

    if not np.allclose(cube1["atom_coords_bohr"], cube2["atom_coords_bohr"], atol=1e-8):
        raise ValueError("hole.cub and electron.cub have different atomic coordinates")


def voxel_volume_bohr3(axes_bohr: np.ndarray) -> float:
    """
    Volume of one voxel = |a · (b × c)|
    where a,b,c are grid increment vectors.
    """
    a, b, c = axes_bohr
    return abs(np.dot(a, np.cross(b, c)))


def build_grid_coordinates_bohr(origin_bohr: np.ndarray, dims: Tuple[int, int, int], axes_bohr: np.ndarray):
    """
    Build all grid-point coordinates in Bohr.
    Returns array of shape (N,3), N=nx*ny*nz
    """
    nx, ny, nz = dims
    ix = np.arange(nx, dtype=float)
    iy = np.arange(ny, dtype=float)
    iz = np.arange(nz, dtype=float)

    # Use meshgrid in ij indexing so it matches cube data ordering
    I, J, K = np.meshgrid(ix, iy, iz, indexing="ij")

    coords = (
        origin_bohr[None, None, None, :]
        + I[..., None] * axes_bohr[0][None, None, None, :]
        + J[..., None] * axes_bohr[1][None, None, None, :]
        + K[..., None] * axes_bohr[2][None, None, None, :]
    )

    return coords.reshape(-1, 3)


def min_distances_to_group(points: np.ndarray, atom_coords: np.ndarray, chunk_size: int = 50000) -> np.ndarray:
    """
    Compute minimum distance from each point to a group of atoms.
    Uses chunking to avoid huge memory use.

    points: (N,3)
    atom_coords: (M,3)

    returns: (N,)
    """
    if atom_coords.shape[0] == 0:
        raise ValueError("Empty atom group encountered.")

    out = np.empty(points.shape[0], dtype=float)

    for start in range(0, points.shape[0], chunk_size):
        end = min(start + chunk_size, points.shape[0])
        p = points[start:end]  # (C,3)
        diff = p[:, None, :] - atom_coords[None, :, :]   # (C,M,3)
        d2 = np.sum(diff * diff, axis=2)                 # (C,M)
        out[start:end] = np.sqrt(np.min(d2, axis=1))

    return out


def integrate_regions(
    hole_data: np.ndarray,
    electron_data: np.ndarray,
    grid_coords_bohr: np.ndarray,
    red_atom_coords_bohr: np.ndarray,
    blue_atom_coords_bohr: np.ndarray,
    dv_bohr3: float,
    cutoff_ang: float = 2.5,
    chunk_size: int = 50000,
):
    """
    Assign each grid point to red or blue by nearest-group rule, with optional cutoff.

    Only points satisfying min(d_red, d_blue) < cutoff are counted in either region.
    Points outside cutoff are ignored for regional decomposition.

    Returns a dict with region integrals and fractions.
    """
    cutoff_bohr = cutoff_ang / BOHR_TO_ANG if cutoff_ang is not None and cutoff_ang > 0 else None

    d_red = min_distances_to_group(grid_coords_bohr, red_atom_coords_bohr, chunk_size=chunk_size)
    d_blue = min_distances_to_group(grid_coords_bohr, blue_atom_coords_bohr, chunk_size=chunk_size)

    nearest_is_red = d_red < d_blue
    nearest_is_blue = ~nearest_is_red

    if cutoff_bohr is not None:
        within_cutoff = np.minimum(d_red, d_blue) < cutoff_bohr
    else:
        within_cutoff = np.ones_like(d_red, dtype=bool)

    mask_red = nearest_is_red & within_cutoff
    mask_blue = nearest_is_blue & within_cutoff

    hole_flat = hole_data.reshape(-1)
    electron_flat = electron_data.reshape(-1)

    # Total density over all grid points
    Qh_all = np.sum(hole_flat) * dv_bohr3
    Qe_all = np.sum(electron_flat) * dv_bohr3

    # Regional density
    Qh_R = np.sum(hole_flat[mask_red]) * dv_bohr3
    Qh_B = np.sum(hole_flat[mask_blue]) * dv_bohr3

    Qe_R = np.sum(electron_flat[mask_red]) * dv_bohr3
    Qe_B = np.sum(electron_flat[mask_blue]) * dv_bohr3

    # Fraction of total hole/electron in each region
    Ph_R = Qh_R / Qh_all if abs(Qh_all) > 1e-15 else np.nan
    Ph_B = Qh_B / Qh_all if abs(Qh_all) > 1e-15 else np.nan

    Pe_R = Qe_R / Qe_all if abs(Qe_all) > 1e-15 else np.nan
    Pe_B = Qe_B / Qe_all if abs(Qe_all) > 1e-15 else np.nan

    # Coverage diagnostics
    Qh_region = Qh_R + Qh_B
    Qe_region = Qe_R + Qe_B

    hole_covered = Qh_region / Qh_all if abs(Qh_all) > 1e-15 else np.nan
    electron_covered = Qe_region / Qe_all if abs(Qe_all) > 1e-15 else np.nan

    return {
        "Qh_all": Qh_all,
        "Qe_all": Qe_all,
        "Qh_R": Qh_R,
        "Qh_B": Qh_B,
        "Qe_R": Qe_R,
        "Qe_B": Qe_B,
        "Ph_R": Ph_R,
        "Ph_B": Ph_B,
        "Pe_R": Pe_R,
        "Pe_B": Pe_B,
        "S_target": Ph_R * Pe_B if not np.isnan(Ph_R) and not np.isnan(Pe_B) else np.nan,
        "S_mismatch": Ph_B * Pe_R if not np.isnan(Ph_B) and not np.isnan(Pe_R) else np.nan,
        "hole_covered": hole_covered,
        "electron_covered": electron_covered,
        "n_red_points": int(np.sum(mask_red)),
        "n_blue_points": int(np.sum(mask_blue)),
        "n_total_points": int(len(mask_red)),
    }


def format_atom_list(atom_list: List[int]) -> str:
    return ",".join(str(x) for x in atom_list)


def main():
    parser = argparse.ArgumentParser(
        description="Integrate hole/electron cube densities over two atom-defined regions."
    )
    parser.add_argument("--hole", required=True, help="Path to hole.cub")
    parser.add_argument("--electron", required=True, help="Path to electron.cub")
    parser.add_argument(
        "--red",
        required=True,
        help='Red region atom indices, 1-based, e.g. "1-12,25-30"',
    )
    parser.add_argument(
        "--blue",
        required=True,
        help='Blue region atom indices, 1-based, e.g. "13-24,31-40"',
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=2.5,
        help="Distance cutoff in Angstrom for assigning grid points to regions. Default: 2.5",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=50000,
        help="Chunk size for distance calculation. Reduce if memory is tight. Default: 50000",
    )

    args = parser.parse_args()

    red_atoms_1based = parse_atom_range(args.red)
    blue_atoms_1based = parse_atom_range(args.blue)

    if len(red_atoms_1based) == 0 or len(blue_atoms_1based) == 0:
        raise ValueError("Red and blue atom lists must not be empty.")

    overlap = set(red_atoms_1based) & set(blue_atoms_1based)
    if overlap:
        raise ValueError(f"Red and blue atom lists overlap: {sorted(overlap)}")

    hole_cube = read_cube(args.hole)
    electron_cube = read_cube(args.electron)
    check_compatibility(hole_cube, electron_cube)

    natoms = hole_cube["natoms"]

    for idx in red_atoms_1based + blue_atoms_1based:
        if idx < 1 or idx > natoms:
            raise ValueError(f"Atom index {idx} out of range. Valid range is 1..{natoms}")

    # Convert 1-based to 0-based
    red_idx = np.array([i - 1 for i in red_atoms_1based], dtype=int)
    blue_idx = np.array([i - 1 for i in blue_atoms_1based], dtype=int)

    atom_coords_bohr = hole_cube["atom_coords_bohr"]
    red_atom_coords_bohr = atom_coords_bohr[red_idx]
    blue_atom_coords_bohr = atom_coords_bohr[blue_idx]

    dv_bohr3 = voxel_volume_bohr3(hole_cube["axes_bohr"])
    grid_coords_bohr = build_grid_coordinates_bohr(
        hole_cube["origin_bohr"], hole_cube["dims"], hole_cube["axes_bohr"]
    )

    result = integrate_regions(
        hole_data=hole_cube["data"],
        electron_data=electron_cube["data"],
        grid_coords_bohr=grid_coords_bohr,
        red_atom_coords_bohr=red_atom_coords_bohr,
        blue_atom_coords_bohr=blue_atom_coords_bohr,
        dv_bohr3=dv_bohr3,
        cutoff_ang=args.cutoff,
        chunk_size=args.chunk,
    )

    print("=" * 72)
    print("Hole/Electron regional integration from cube files")
    print("=" * 72)
    print(f"Hole cube      : {args.hole}")
    print(f"Electron cube  : {args.electron}")
    print(f"Grid dims      : {hole_cube['dims']}")
    print(f"Voxel volume   : {dv_bohr3:.8e} bohr^3")
    print(f"Red atoms      : {format_atom_list(red_atoms_1based)}")
    print(f"Blue atoms     : {format_atom_list(blue_atoms_1based)}")
    print(f"Cutoff         : {args.cutoff:.3f} Å")
    print("-" * 72)

    print("Regional fractions:")
    print(f"Ph(R)          = {result['Ph_R']:.6f}")
    print(f"Ph(B)          = {result['Ph_B']:.6f}")
    print(f"Pe(R)          = {result['Pe_R']:.6f}")
    print(f"Pe(B)          = {result['Pe_B']:.6f}")
    print("-" * 72)
    print("Composite scores:")
    print(f"S_target       = Ph(R) * Pe(B) = {result['S_target']:.6f}")
    print(f"S_mismatch     = Ph(B) * Pe(R) = {result['S_mismatch']:.6f}")
    print("-" * 72)
    print("Coverage diagnostics:")
    print(f"Hole covered   = {result['hole_covered']:.6f}")
    print(f"Electron covered = {result['electron_covered']:.6f}")
    print("  (If coverage is too low, increase cutoff or reconsider region definition.)")
    print("-" * 72)
    print("Raw integrals:")
    print(f"Qh(all)        = {result['Qh_all']:.8e}")
    print(f"Qh(R)          = {result['Qh_R']:.8e}")
    print(f"Qh(B)          = {result['Qh_B']:.8e}")
    print(f"Qe(all)        = {result['Qe_all']:.8e}")
    print(f"Qe(R)          = {result['Qe_R']:.8e}")
    print(f"Qe(B)          = {result['Qe_B']:.8e}")
    print("-" * 72)
    print("Grid point counts:")
    print(f"Red points     = {result['n_red_points']}")
    print(f"Blue points    = {result['n_blue_points']}")
    print(f"Total points   = {result['n_total_points']}")
    print("=" * 72)

    # A brief interpretation
    print("\nInterpretation:")
    if result["Ph_R"] > result["Ph_B"] and result["Pe_B"] > result["Pe_R"]:
        print("Hole is mainly in RED region; electron is mainly in BLUE region.")
    elif result["Ph_B"] > result["Ph_R"] and result["Pe_R"] > result["Pe_B"]:
        print("Hole is mainly in BLUE region; electron is mainly in RED region.")
    else:
        print("Hole/electron are not cleanly separated into the expected RED/BLUE pattern.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
