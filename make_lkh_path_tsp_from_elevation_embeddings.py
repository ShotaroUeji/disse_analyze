#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_lkh_path_tsp_from_elevation_embeddings.py

Create an LKH-compatible TSP file for elevation embeddings using a dummy node.

Purpose:
  Elevation is a linear variable, not circular.
  Therefore, we want a Hamiltonian path, not a TSP cycle.

Method:
  Add one dummy node.
  Distance(dummy, every real node) = 0.
  Solve TSP cycle with LKH.
  Remove dummy from the tour to obtain a Hamiltonian path.

Input:
  continuous_elevation_embeddings.pt

Expected pt format:
  {
    "elevations_deg": Tensor[N],
    "embeddings": {
      "audio_space": Tensor[N, D],
      "audio_source": Tensor[N, D],
      ...
    }
  }

Output:
  - .tsp file for LKH
  - .labels.tsv file mapping node ids to elevation values
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def get_sweep_values(obj):
    if "elevations_deg" in obj:
        return obj["elevations_deg"].detach().cpu().numpy().astype(float)
    if "angles_deg" in obj:
        # fallback for compatibility
        return obj["angles_deg"].detach().cpu().numpy().astype(float)
    raise KeyError("The .pt file must contain 'elevations_deg' or 'angles_deg'.")


def pairwise_wordtour_distance(Z: torch.Tensor, normalize: bool = True) -> np.ndarray:
    """
    WordTour-style distance:
      Euclidean distance between embedding vectors.
    Edge weights are later written as int(distance * 1000).
    """
    Z = Z.float().cpu()
    if normalize:
        Z = F.normalize(Z, dim=-1)

    X = Z.numpy().astype(np.float64)
    diff = X[:, None, :] - X[None, :, :]
    D = np.sqrt(np.sum(diff * diff, axis=-1))
    W = np.floor(D * 1000.0).astype(np.int64)
    np.fill_diagonal(W, 0)
    return W


def write_lkh_tsp_with_dummy(W: np.ndarray, out_tsp: Path, name: str):
    """
    Write TSPLIB EXPLICIT UPPER_ROW format.

    Real nodes: 1..N
    Dummy node: N+1
    Distance(dummy, any real node) = 0
    """
    n = W.shape[0]
    dim = n + 1
    dummy_id = dim

    out_tsp.parent.mkdir(parents=True, exist_ok=True)

    with out_tsp.open("w", encoding="utf-8") as f:
        f.write(f"NAME: {name}\n")
        f.write("TYPE: TSP\n")
        f.write(f"DIMENSION: {dim}\n")
        f.write("EDGE_WEIGHT_TYPE: EXPLICIT\n")
        f.write("EDGE_WEIGHT_FORMAT: UPPER_ROW\n")
        f.write("EDGE_WEIGHT_SECTION\n")

        for i in range(1, dim + 1):
            vals = []
            for j in range(i + 1, dim + 1):
                if i == dummy_id or j == dummy_id:
                    vals.append("0")
                else:
                    vals.append(str(int(W[i - 1, j - 1])))
            if vals:
                f.write(" ".join(vals) + "\n")
            else:
                f.write("\n")

        f.write("EOF\n")


def write_labels(elevations: np.ndarray, out_labels: Path):
    """
    LKH node ids are 1-based.
    Real nodes: 1..N
    Dummy node: N+1
    """
    out_labels.parent.mkdir(parents=True, exist_ok=True)

    n = len(elevations)
    dummy_id = n + 1

    with out_labels.open("w", encoding="utf-8") as f:
        f.write("node_id\tlabel\televation_deg\tis_dummy\n")
        for i, el in enumerate(elevations, start=1):
            label = f"el{float(el):+.3f}"
            f.write(f"{i}\t{label}\t{float(el):.8f}\t0\n")
        f.write(f"{dummy_id}\tdummy\tnan\t1\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb_pt", required=True)
    p.add_argument("--key", required=True,
                   choices=["audio_space", "audio_source", "text_space", "text_source"])
    p.add_argument("--out_tsp", required=True)
    p.add_argument("--out_labels", required=True)
    p.add_argument("--name", default=None)
    p.add_argument("--no_normalize", action="store_true")
    args = p.parse_args()

    obj = torch.load(args.emb_pt, map_location="cpu")
    elevations = get_sweep_values(obj)

    if "embeddings" not in obj:
        raise KeyError("The .pt file must contain 'embeddings'.")

    if args.key not in obj["embeddings"]:
        raise KeyError(f"Embedding key not found: {args.key}")

    Z = obj["embeddings"][args.key]
    if len(elevations) != Z.shape[0]:
        raise ValueError(f"N mismatch: elevations={len(elevations)}, embeddings={Z.shape[0]}")

    W = pairwise_wordtour_distance(Z, normalize=not args.no_normalize)

    out_tsp = Path(args.out_tsp)
    out_labels = Path(args.out_labels)
    name = args.name or out_tsp.stem

    write_lkh_tsp_with_dummy(W, out_tsp, name=name)
    write_labels(elevations, out_labels)

    print(f"[✓] wrote TSP:    {out_tsp}")
    print(f"[✓] wrote labels: {out_labels}")
    print(f"[info] key={args.key}, N={Z.shape[0]}, dim={Z.shape[1]}")
    print(f"[info] dummy node id = {Z.shape[0] + 1}")
    print("[info] distance = int(1000 * Euclidean(L2-normalized embedding distance))")


if __name__ == "__main__":
    main()
