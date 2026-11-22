# -*- coding: utf-8 -*-
"""DeepShip mainline runner (unified zero-contact pool, six spaces).

The DeepShip mainline is the closed-form evaluation of the class model:
the injection weight is zero there (direction alignment is negative), so the
sampled readout coincides with the exact evaluation, a per-space PCA(256) +
shrinkage-LDA classifier whose equal-weight pool is the paper's 78.25.
Use --sampled for the synthesis-pipeline readout (~77.7), which draws from
the same class model instead of evaluating it exactly.

Each pool space is `<name>.npz` with E_tr, y_tr, E_te, y_te, Tx, labs.

Usage:
  python runners/run_deepship.py --emb-dir embeddings/deepship \
      --spaces clap_l clap_m da_clap_l ssl_cqt_1 ssl_cqt_2 ssl_cqt_3p
"""
import argparse
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tadc.core import LAM, closed_form_readout, fit_readout, rule_w, support_alpha

DS_CLS = ["0", "1", "2", "3"]  # cargo / passenger / tanker / tug
N_PER, PCA_D, S0 = 16, 256, 0.5  # sampled-readout configuration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--spaces", nargs="+",
                    default=["clap_l", "clap_m", "da_clap_l",
                             "ssl_cqt_1", "ssl_cqt_2", "ssl_cqt_3p"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--sampled", action="store_true",
                    help="synthesis-pipeline readout instead of the closed form")
    args = ap.parse_args()

    zs = {n: np.load(Path(args.emb_dir) / f"{n}.npz", allow_pickle=True)
          for n in args.spaces}
    y_te = zs[args.spaces[0]]["y_te"].astype(str)

    alpha = support_alpha(zs["clap_l"]["E_tr"], zs["clap_l"]["y_tr"].astype(str),
                          zs["clap_l"]["Tx"], zs["clap_l"]["labs"].astype(str),
                          DS_CLS)
    w = rule_w(alpha, LAM)
    print(f"direction alignment {alpha:+.4f} -> w = {w:.4f}")

    if not args.sampled:
        PK_sum = None
        for n in args.spaces:
            z = zs[n]
            predict = closed_form_readout(z["E_tr"], z["y_tr"].astype(str),
                                          DS_CLS, pca_d=PCA_D)
            PK = predict(z["E_te"])
            PK_sum = PK if PK_sum is None else PK_sum + PK
        acc = 100 * float(np.mean(np.array(DS_CLS)[PK_sum.argmax(1)] == y_te))
        print(f"DeepShip mainline (closed form): {acc:.2f}  (paper: 78.25)")
        return

    accs = []
    for sd in args.seeds:
        PK_sum = None
        for n in args.spaces:
            z = zs[n]
            predict = fit_readout(z["E_tr"], z["y_tr"].astype(str), z["Tx"],
                                  z["labs"].astype(str), DS_CLS, seed=sd, w=w,
                                  s=S0, n_per=N_PER, pca_d=PCA_D)
            PK = predict(z["E_te"])
            PK_sum = PK if PK_sum is None else PK_sum + PK
        accs.append(100 * float(np.mean(np.array(DS_CLS)[PK_sum.argmax(1)] == y_te)))
        print(f"seed {sd}: pool acc = {accs[-1]:.2f}")
    print(f"DeepShip sampled readout: {np.mean(accs):.2f} +/- {np.std(accs):.2f} "
          f"(paper sampled: ~77.7)")


if __name__ == "__main__":
    main()
