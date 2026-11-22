# -*- coding: utf-8 -*-
"""IARA mainline runner (unified zero-contact pool, six spaces, official
5x2 cross-validation, recording-level voting, balanced accuracy).

The direction alignment is negative on IARA (~ -0.02), so the rule sets
w = 0 and text enters only the alignment measurement, never the fit.

Embeddings `<name>.npz` carry E (all windows), y (0-3), iid (recording id),
and, for the self-supervised spaces, m (window-validity mask); Tx/labs are
the 80-sentence library embedded with the space's text tower. Splits follow
the official protocol (splits/iara_recordings.csv), reproducing the ten
folds; the three synthesis seeds follow the paper.

Usage:
  python runners/run_iara.py --emb-dir embeddings/iara \
      --spaces clap_l clap_m da_clap_l ssl_cqt_1 ssl_cqt_2 ssl_cqt_3p
"""
import argparse
import collections
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tadc.core import LAM, fit_readout, rule_w, support_alpha

CLS4 = ["SMALL", "MEDIUM", "LARGE", "BG"]
N_PER, PCA_D, S0 = 48, 256, 0.5
RNG_SEEDS = 3


def official_splits(df_ids, df_targets):
    splits = []
    for i in range(5):
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42 * i)
        for a, b in skf.split(df_ids, df_targets):
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25,
                                         random_state=42 * i)
            vt = b
            ti, vi = next(iter(sss.split(np.zeros(len(vt)), df_targets[vt])))
            splits.append((a, vt[vi], vt[ti]))
    return splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--spaces", nargs="+",
                    default=["clap_l", "clap_m", "da_clap_l",
                             "ssl_cqt_1", "ssl_cqt_2", "ssl_cqt_3p"])
    args = ap.parse_args()

    zs = {n: np.load(Path(args.emb_dir) / f"{n}.npz", allow_pickle=True)
          for n in args.spaces}
    iid = zs[args.spaces[0]]["iid"].astype(int)
    y_all = np.array([CLS4[int(v)] for v in zs[args.spaces[0]]["y"]])
    n = len(iid)
    masks = {k: (z["m"].astype(bool) if "m" in z.files else np.ones(n, bool))
             for k, z in zs.items()}

    import pandas as pd
    csv = Path(__file__).resolve().parents[1] / "splits" / "iara_recordings.csv"
    df = pd.read_csv(csv).drop_duplicates("ID")
    tgt = lambda l: 3 if pd.isna(l) else (0 if l < 50 else (1 if l < 100 else 2))
    df["t"] = df["Length"].apply(tgt).values
    ids = df["ID"].astype(int).values
    t = df["t"].values
    ymap = {int(r): CLS4.index(c) for c, r in zip(y_all, iid)}

    fold_auds, fold_alphas, seed_scores = [], [], [[] for _ in range(RNG_SEEDS)]
    for fold, (a, _, te_idx) in enumerate(official_splits(ids, t)):
        tr_ids = set(ids[a].tolist())
        te_ids = set(ids[te_idx].tolist())
        trm = np.array([r in tr_ids for r in iid])
        tem = np.array([r in te_ids for r in iid])
        tem_pos = np.where(tem)[0]
        alpha = support_alpha(zs["clap_l"]["E"][trm & masks["clap_l"]],
                              y_all[trm & masks["clap_l"]],
                              zs["clap_l"]["Tx"], zs["clap_l"]["labs"].astype(str),
                              CLS4)
        w = rule_w(alpha, LAM)
        fold_alphas.append(alpha)
        aud_scores = []
        for sd in range(RNG_SEEDS):
            PKs = np.zeros((tem.sum(), 4))
            for name in args.spaces:
                E, m = zs[name]["E"], masks[name]
                cm = trm & m
                if cm.sum() < 4 or len(set(y_all[cm])) < 2:
                    continue
                predict = fit_readout(E[cm], y_all[cm], zs[name]["Tx"],
                                      zs[name]["labs"].astype(str), CLS4,
                                      seed=1000 * sd + fold, w=w, s=S0,
                                      n_per=N_PER, pca_d=PCA_D)
                PKs[m[tem_pos]] = PKs[m[tem_pos]] + predict(E[tem & m])
            pr = PKs.argmax(1)
            votes = collections.defaultdict(list)
            for p_, r in zip(pr, iid[tem]):
                votes[r].append(int(p_))
            pf = np.array([collections.Counter(v).most_common(1)[0][0]
                           for v in votes.values()])
            yf = np.array([ymap[k] for k in votes])
            s = 100 * balanced_accuracy_score(yf, pf)
            aud_scores.append(s)
            seed_scores[sd].append(s)
        fold_auds.append(float(np.mean(aud_scores)))
    # the paper reports mean +/- std over synthesis seeds, each seed averaged
    # over the ten official folds
    seed_means = [float(np.mean(v)) for v in seed_scores]
    print(f"IARA mainline: {np.mean(seed_means):.2f} +/- {np.std(seed_means):.2f} "
          f"(paper: 70.03 +/- 0.01; alpha ~ {np.mean(fold_alphas):+.4f} -> w = "
          f"{rule_w(float(np.mean(fold_alphas))):.4f}; seeds "
          f"{['%.2f' % v for v in seed_means]})")


if __name__ == "__main__":
    main()
