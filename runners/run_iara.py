# -*- coding: utf-8 -*-
"""IARA mainline runner (3 CLAP-family spaces, official 5x2 cross-validation,
recording-level voting, balanced accuracy; gate off: w=0).

Embeddings `<name>.npz` additionally carry `iid` (recording id per window);
splits follow the official protocol (iara.csv), reproducing the 10 folds.
Mainline text prefix = 200 LLM sentences per class + 40 template repeats.

Usage:
  python runners/run_iara.py --emb-dir embeddings/iara \
      --spaces clap_l clap_m da_clap_l
"""
import argparse
import collections
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tadc.core import fit_readout

CLS4 = ["SMALL", "MEDIUM", "LARGE", "BG"]
N_PER, PCA_D, S0 = 48, 256, 0.5  # w=0 by the gate (alignment -0.02)
RNG_SEEDS = 5


def official_splits(df_ids, df_targets):
    splits = []
    for i in range(5):
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42 * i)
        for a, b in skf.split(df_ids, df_targets):
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42 * i)
            vt = b
            ti, vi = next(iter(sss.split(np.zeros(len(vt)), df_targets[vt])))
            splits.append((a, vt[vi], vt[ti]))
    return splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--spaces", nargs="+", default=["clap_l", "clap_m", "da_clap_l"])
    args = ap.parse_args()

    # one space provides iid + y; all spaces share the window grid
    zs = {n: np.load(Path(args.emb_dir) / f"{n}.npz",
                     allow_pickle=True) for n in args.spaces}
    iid = zs[args.spaces[0]]["iid"].astype(int)
    y_all = zs[args.spaces[0]]["y"].astype(str)

    # recording-level table (length grades come with the IARA distribution,
    # iara.csv: ID,Length; target = 0 (<50m) / 1 (<100m) / 2 (>=100m) / 3 BG)
    import pandas as pd
    csv = Path(__file__).resolve().parents[1] / "splits" / "iara_recordings.csv"
    df = pd.read_csv(csv)
    df = df.drop_duplicates("ID")
    tgt = lambda l: 3 if pd.isna(l) else (0 if l < 50 else (1 if l < 100 else 2))
    df["t"] = df["Length"].apply(tgt).values
    ids = df["ID"].astype(int).values
    t = df["t"].values

    fold_auds = []
    for fold, (a, _, _) in enumerate(official_splits(ids, t)):
        tr_ids = set(ids[a].tolist())
        trm = np.array([r in tr_ids for r in iid])
        aud_scores = []
        for rsi in range(RNG_SEEDS):
            sd = 1000 * rsi + fold
            PKsum = None
            for n in args.spaces:
                z = zs[n]
                E, yy = z["E"], z["y"].astype(str)
                predict = fit_readout(E[trm], yy[trm], z["Tx"], z["labs"], CLS4,
                                      seed=sd, w=0.0, s=S0, n_per=N_PER,
                                      pca_d=PCA_D, fam="gauss")
                PK = predict(E)
                PKsum = PK if PKsum is None else PKsum + PK
            pr = PKsum.argmax(1)
            votes = collections.defaultdict(list)
            for p_, r in zip(pr, iid):
                votes[r].append(int(p_))
            ymap = {int(r): int(v) for v, r in zip(
                np.array([CLS4.index(c) for c in y_all]), iid)}
            pf = np.array([collections.Counter(v).most_common(1)[0][0]
                           for v in votes.values()])
            yf = np.array([ymap[k] for k in votes])
            aud_scores.append(100 * balanced_accuracy_score(yf, pf))
        fold_auds.append(np.mean(aud_scores))
    print(f"IARA mainline: {np.mean(fold_auds):.2f} +/- {np.std(fold_auds):.2f}")


if __name__ == "__main__":
    main()
