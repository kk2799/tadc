# -*- coding: utf-8 -*-
"""TADC core: text-anchored distribution calibration in a frozen feature space.

Implements the paper's Eq. (1)-(2): each synthetic sample is
    p = b_c + w * (t_j - t_bar) + s * L_c eps,   eps ~ N(0, I)
where b_c is the class base (class mean or a k-means draw), (t_j - t_bar) is the
centered text anchor of class c, and L_c is the Cholesky factor of the pooled
within-class covariance. A PCA + logistic-regression readout is fitted on the
synthetic set only. The gate sets w = w0 when the direction alignment
alpha > 0 and w = 0 otherwise.

Dependencies: numpy, scikit-learn.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def class_bases(E_tr, y_tr, groups, fam="gauss"):
    """Class base b_c. fam='gauss' uses the class mean; fam='kmeans' uses
    per-class k-means centers with mixture weights (used on DeepShip)."""
    d_ = E_tr.shape[1]
    yidx = np.asarray([groups.index(y) for y in y_tr]) if isinstance(y_tr[0], str) \
        else np.asarray(y_tr)
    if fam == "gauss":
        mus = np.stack([E_tr[yidx == i].mean(0) for i in range(len(groups))])
        return mus, None
    cl_ = {}
    for i in range(len(groups)):
        X2 = E_tr[yidx == i]
        if len(X2) < 2:
            continue
        Km = max(1, min(5, len(X2) // 3))
        km = KMeans(Km, n_init=3, random_state=0).fit(X2)
        cl_[i] = (km.cluster_centers_, np.bincount(km.labels_, minlength=Km) / len(X2))
    return cl_, None


def pooled_cholesky(E_tr, y_tr, groups):
    d_ = E_tr.shape[1]
    yidx = np.asarray([groups.index(y) for y in y_tr]) if isinstance(y_tr[0], str) \
        else np.asarray(y_tr)
    mus = np.stack([E_tr[yidx == i].mean(0) for i in range(len(groups))])
    Xc = np.concatenate([E_tr[yidx == i] - mus[i] for i in range(len(groups))])
    Sg = Xc.T @ Xc / max(len(Xc) - len(groups), 1) + 1e-6 * np.eye(d_)
    return np.linalg.cholesky(Sg), mus


def fit_readout(E_tr, y_tr, Tx, labs, groups, seed=0, w=0.0, s=0.5,
                n_per=16, pca_d=256, fam="gauss"):
    """One TADC fit in one space. Returns (PK on any embedding batch, pca_dim)."""
    d_ = E_tr.shape[1]
    yidx = np.asarray([groups.index(y) for y in y_tr]) if isinstance(y_tr[0], str) \
        else np.asarray(y_tr)
    gi = np.asarray([groups.index(c) for c in labs])
    Lc, mus = pooled_cholesky(E_tr, y_tr, groups)
    cl_ = None
    if fam == "kmeans":
        cl_ = {}
        for i in range(len(groups)):
            X2 = E_tr[yidx == i]
            if len(X2) < 2:
                continue
            Km = max(1, min(5, len(X2) // 3))
            km = KMeans(Km, n_init=3, random_state=0).fit(X2)
            cl_[i] = (km.cluster_centers_, np.bincount(km.labels_, minlength=Km) / len(X2))
    t_bar = Tx.mean(0)
    rng = np.random.default_rng(seed)
    rows, lr_ = [], []
    for k in range(len(Tx)):
        if fam == "kmeans" and gi[k] in cl_:
            cs, pw = cl_[gi[k]]
            base = cs[rng.choice(len(cs), p=pw)]
        else:
            base = mus[gi[k]]
        shift = w * (Tx[k] - t_bar)
        for _ in range(n_per):
            rows.append(base + shift + s * (rng.standard_normal(d_) @ Lc.T))
            lr_.append(labs[k])
    P = np.asarray(rows)
    P = P / np.clip(np.linalg.norm(P, axis=1, keepdims=True), 1e-8, None)
    p_ = PCA(pca_d, random_state=0).fit(P)
    sclr = StandardScaler().fit(p_.transform(P))
    clf = LogisticRegression(max_iter=1000).fit(sclr.transform(p_.transform(P)), np.asarray(lr_))

    def predict(E):
        pe = clf.predict_proba(sclr.transform(p_.transform(E)))
        PK = np.zeros((len(E), len(groups)))
        for j, c in enumerate(clf.classes_):
            PK[:, groups.index(c)] += pe[:, j]
        return PK

    return predict


def direction_alignment(E_tr, y_tr, Tx, labs, groups):
    """Gate statistic alpha: mean cosine between centered anchors and the
    audio class-difference directions. Computed before any fitting."""
    yidx = np.asarray([groups.index(y) for y in y_tr]) if isinstance(y_tr[0], str) \
        else np.asarray(y_tr)
    mus = np.stack([E_tr[yidx == i].mean(0) for i in range(len(groups))])
    t_bar = Tx.mean(0)
    num, den = [], []
    for c in range(len(groups)):
        m = np.asarray([groups.index(x) for x in labs])
        Tc = Tx[m == c] - t_bar
        gm = mus[c] - mus.mean(0)
        for t in Tc:
            n = float(t @ gm)
            num.append(n)
            den.append(float(np.linalg.norm(t) * np.linalg.norm(gm)))
    return float(np.sum(num) / (np.sum(den) + 1e-12))


def gated_w(alpha, w0=0.02):
    """The rule of the paper: inject only where alignment is positive."""
    return w0 if alpha > 0 else 0.0


def pool_predict(predicts, E):
    """Ensemble of spaces: equal-weight average of per-space probabilities."""
    return np.mean([p(E) for p in predicts], axis=0)
