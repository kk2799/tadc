# -*- coding: utf-8 -*-
"""ShipsEar mainline runner (7 spaces, gate on: w=0.02, gauss base).

Each pool space is `<name>.npz` with E_tr, y_tr, E_te, y_te, Tx, labs
(text-library embeddings via that space's text tower, mainline prefix = 40
LLM sentences per type + 40 template repeats; templates in data/text/).

Usage:
  python runners/run_shipsear.py --emb-dir embeddings/shipsear \
      --spaces clap_l da_clap_l da_clap_m_a da_clap_m_b da_htsat da_beats proj_ast
"""
import argparse
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tadc.core import fit_readout, direction_alignment, gated_w

N_PER, PCA_D, S0, W0 = 48, 128, 0.35, 0.02  # mainline config (factorial_SE)

# ShipsEar: 9 ship types -> 5 aggregated classes (the paper's Table mapping)
TYPE2CLS = {"Fishboat": "A", "Musselboat": "A", "Dredger": "A",
            "Motorboat": "B", "Sailboat": "B",
            "Passengers": "C",
            "Oceanliner": "D", "RORO": "D",
            "Naturalambientnoise": "E"}


def classes_of(y):
    return sorted(set(y.tolist()))


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
            labs_cls = np.array([TYPE2CLS.get(l, l) for l in labs.astype(str)])
            groups = classes_of(y_tr)
            alpha = direction_alignment(E_tr, y_tr, Tx, labs_cls, groups)
            w = gated_w(alpha, W0)
            predict = fit_readout(E_tr, y_tr, Tx, labs_cls, groups, seed=sd, w=w,
                                  s=S0, n_per=N_PER, pca_d=PCA_D, fam="gauss")
            PK = predict(E_te)
            PK_sum = PK if PK_sum is None else PK_sum + PK
        acc = float(np.mean(np.array(classes_of(y_te))[PK_sum.argmax(1)] == y_te))
        accs.append(100 * acc)
        print(f"seed {sd}: pool acc = {100*acc:.2f}")
    print(f"ShipsEar mainline: {np.mean(accs):.2f} +/- {np.std(accs):.2f}")


if __name__ == "__main__":
    main()
