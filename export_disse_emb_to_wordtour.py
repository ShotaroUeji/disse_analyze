#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export DISSE continuous azimuth embeddings to a WordTour/GloVe-like text file.

Input:
  continuous_azimuth_embeddings.pt

Output:
  embeddings_audio_space.txt
  labels_audio_space.txt

The embedding file format is:
  az000 v1 v2 ...
  az005 v1 v2 ...
  ...
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb_pt", required=True)
    p.add_argument("--key", default="audio_space",
                   choices=["audio_space", "audio_source", "text_space", "text_source"])
    p.add_argument("--out_txt", required=True)
    p.add_argument("--out_labels", default=None)
    p.add_argument("--normalize", action="store_true")
    args = p.parse_args()

    obj = torch.load(args.emb_pt, map_location="cpu")
    angles = obj["angles_deg"].cpu().numpy()
    Z = obj["embeddings"][args.key].float().cpu()

    if args.normalize:
        Z = F.normalize(Z, dim=-1)

    out_txt = Path(args.out_txt)
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    if args.out_labels is None:
        out_labels = out_txt.with_suffix(".labels.txt")
    else:
        out_labels = Path(args.out_labels)

    with out_txt.open("w", encoding="utf-8") as f, out_labels.open("w", encoding="utf-8") as g:
        for i, az in enumerate(angles):
            # WordTour treats the first token as a word.
            # Use zero-padded positive degree labels.
            label = f"az{int(round(float(az))) % 360:03d}"
            vals = " ".join(f"{x:.8f}" for x in Z[i].tolist())
            f.write(f"{label} {vals}\n")
            g.write(f"{i+1}\t{label}\t{float(az):.6f}\n")

    print(f"[✓] wrote {out_txt}")
    print(f"[✓] wrote {out_labels}")
    print(f"[info] key={args.key}, n={Z.shape[0]}, dim={Z.shape[1]}")


if __name__ == "__main__":
    main()