# -*- coding: utf-8 -*-
"""ShipsEar mainline runner (unified zero-contact pool, six spaces).

Fitting and prediction are at the nine ship-type level; predictions are then
mapped to the five aggregated classes A-E for scoring. The injection weight
is set by the paper's rule w = lam * max(alpha, 0) with lam = 0.40, where
alpha is the direction alignment measured on the labeled support in the
public CLAP-L space (audio class-mean differences vs. template class-mean
differences). On ShipsEar alpha is positive (~ +0.09) and w ~= 0.04.

Each pool space is `<name>.npz` with E_tr, y_tr, E_te, y_te (nine-type
labels), Tx, labs (80 sentences per type: 40 template repeats first, then 40
generated sentences; see backbones/embed_texts.py).

Usage:
  python runners/run_shipsear.py --emb-dir embeddings/shipsear \
      --spaces clap_l clap_m da_clap_l ssl_cqt_1 ssl_cqt_2 ssl_cqt_3p
"""
import argparse
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tadc.core import LAM, fit_readout, rule_w, support_alpha

N_PER, PCA_D, S0 = 96, 128, 0.35  # ShipsEar fit configuration

TYPE2CLS = {"Fishboat": "A", "Musselboat": "A", "Dredger": "A",
            "Motorboat": "B", "Sailboat": "B",
            "Passengers": "C",
            "Oceanliner": "D", "RORO": "D",
            "Naturalambientnoise": "E"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--spaces", nargs="+",
                    default=["clap_l", "clap_m", "da_clap_l",
                             "ssl_cqt_1", "ssl_cqt_2", "ssl_cqt_3p"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    zs = {n: np.load(Path(args.emb_dir) / f"{n}.npz", allow_pickle=True)
          for n in args.spaces}
    groups = sorted({t for t in zs[args.spaces[0]]["y_tr"].astype(str)})

    accs, alphas = [], []
    for sd in args.seeds:
        alpha = support_alpha(zs["clap_l"]["E_tr"], zs["clap_l"]["y_tr"].astype(str),
                              zs["clap_l"]["Tx"], zs["clap_l"]["labs"].astype(str),
                              groups)
        w = rule_w(alpha, LAM)
        PK_sum, y_te = None, None
        for n in args.spaces:
            z = zs[n]
            E_tr, E_te = z["E_tr"], z["E_te"]
            y_tr, y_te = z["y_tr"].astype(str), z["y_te"].astype(str)
            labs = z["labs"].astype(str)
            predict = fit_readout(E_tr, y_tr, z["Tx"], labs, groups,
                                  seed=sd, w=w, s=S0, n_per=N_PER, pca_d=PCA_D)
            PK = predict(E_te)
            PK_sum = PK if PK_sum is None else PK_sum + PK
        pr_type = np.array(groups)[PK_sum.argmax(1)]
        pr5 = np.array([TYPE2CLS.get(p, "A") for p in pr_type])
        y5 = np.array([TYPE2CLS[t] for t in y_te])
        accs.append(100 * float(np.mean(pr5 == y5)))
        alphas.append(alpha)
        print(f"seed {sd}: alpha = {alpha:+.4f}, w = {w:.4f}, "
              f"pool acc = {accs[-1]:.2f}")
    print(f"ShipsEar mainline: {np.mean(accs):.2f} +/- {np.std(accs):.2f} "
          f"(paper: 85.92 +/- 0.41)")


if __name__ == "__main__":
    main()
