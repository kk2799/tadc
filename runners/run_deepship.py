# -*- coding: utf-8 -*-
"""DeepShip mainline runner (greedy six-member pool, w=0 by the gate).

Each pool space is a directory entry `<name>.npz` holding
    E_tr, y_tr : training-block embeddings and labels (str class ids)
    E_te, y_te : test-block embeddings and labels
    Tx, labs   : text-library embeddings (already embedded with this space's
                 text tower) and their class ids
The gate (direction alignment, computed before fitting) sets w; TADC is fitted
per space on synthetic samples only, and the pool is the equal-weight average
of per-space probabilities.

Usage:
  python runners/run_deepship.py --emb-dir embeddings/deepship \
      --spaces clap_l clap_m da_clap_l ssl_cqt_1 ssl_cqt_2 ssl_cqt_3p
"""
import argparse
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tadc.core import fit_readout, direction_alignment, gated_w

DS_CLS = ["0", "1", "2", "3"]  # cargo / passenger / tanker / tug
N_PER, PCA_D, S0, W0 = 16, 256, 0.5, 0.02


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--spaces", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    accs = []
    for sd in args.seeds:
        PK_sum, y_te = None, None
        for name in args.spaces:
            z = np.load(Path(args.emb_dir) / f"{name}.npz", allow_pickle=True)
            E_tr, E_te, Tx, labs = z["E_tr"], z["E_te"], z["Tx"], z["labs"]
            y_tr = z["y_tr"].astype(str)
            y_te = z["y_te"].astype(str)
            alpha = direction_alignment(E_tr, y_tr, Tx, labs, DS_CLS)
            w = gated_w(alpha, W0)
            predict = fit_readout(E_tr, y_tr, Tx, labs, DS_CLS, seed=sd, w=w,
                                  s=S0, n_per=N_PER, pca_d=PCA_D, fam="kmeans")
            PK = predict(E_te)
            PK_sum = PK if PK_sum is None else PK_sum + PK
        acc = float(np.mean(np.array(DS_CLS)[PK_sum.argmax(1)] == y_te))
        accs.append(100 * acc)
        print(f"seed {sd}: pool acc = {100*acc:.2f}")
    print(f"DeepShip mainline: {np.mean(accs):.2f} +/- {np.std(accs):.2f}")


if __name__ == "__main__":
    main()
