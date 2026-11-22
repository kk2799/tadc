# -*- coding: utf-8 -*-
"""TADC core: text-anchored distribution calibration in a frozen feature space.

Implements the paper's synthesis: each synthetic sample is
    p = b_c + w * (t_j - t_bar) + s * L_c eps,   eps ~ N(0, I)
where b_c is the class base (class mean), (t_j - t_bar) is the centered text
anchor of class c, and L_c is the Cholesky factor of the pooled within-class
covariance. A PCA + logistic-regression readout is fitted on the synthetic set
only. Where the injection weight is zero, the same class model also has an
exact (closed-form) evaluation: a PCA + Gaussian (LDA) classifier on the
labeled embeddings, which is what the DeepShip mainline reports.

The injection weight follows the paper's rule
    w = lam * max(alpha, 0),   lam = 0.40 (fixed once for all datasets),
where alpha is the direction alignment: the mean cosine between audio
class-mean difference directions and text class-mean difference directions,
computed on the labeled support before any fitting.

Dependencies: numpy, scikit-learn.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

LAM = 0.40  # the paper's single operating point, fixed across datasets


def _yidx(y, groups):
    return np.asarray([groups.index(y_) for y_ in y]) if isinstance(y[0], str) \
        else np.asarray(y)


def pooled_cholesky(E_tr, y_tr, groups):
    d_ = E_tr.shape[1]
    yidx = _yidx(y_tr, groups)
    mus = np.stack([E_tr[yidx == i].mean(0) for i in range(len(groups))])
    Xc = np.concatenate([E_tr[yidx == i] - mus[i] for i in range(len(groups))])
    Sg = Xc.T @ Xc / max(len(Xc) - len(groups), 1) + 1e-6 * np.eye(d_)
    return np.linalg.cholesky(Sg), mus


def fit_readout(E_tr, y_tr, Tx, labs, groups, seed=0, w=0.0, s=0.5,
                n_per=16, pca_d=256):
    """One TADC fit in one space: synthesize n_per samples per text row,
    then fit PCA + standardized logistic regression. Returns a predictor."""
    d_ = E_tr.shape[1]
    gi = np.asarray([groups.index(c) for c in labs])
    Lc, mus = pooled_cholesky(E_tr, y_tr, groups)
    t_bar = Tx.mean(0)
    rng = np.random.default_rng(seed)
    rows, lr_ = [], []
    for k in range(len(Tx)):
        shift = w * (Tx[k] - t_bar) if w else 0.0
        for _ in range(n_per):
            rows.append(mus[gi[k]] + shift + s * (rng.standard_normal(d_) @ Lc.T))
            lr_.append(gi[k])
    P = np.asarray(rows)
    P = P / np.clip(np.linalg.norm(P, axis=1, keepdims=True), 1e-8, None)
    p_ = PCA(min(pca_d, P.shape[0] - 1, P.shape[1]), random_state=0).fit(P)
    sc = StandardScaler().fit(p_.transform(P))
    clf = LogisticRegression(max_iter=1000).fit(
        sc.transform(p_.transform(P)), np.asarray(lr_))

    def predict(E):
        pe = clf.predict_proba(sc.transform(p_.transform(E)))
        PK = np.zeros((len(E), len(groups)))
        for j, c in enumerate(clf.classes_):
            PK[:, int(c)] += pe[:, j]
        return PK

    return predict


def closed_form_readout(E_tr, y_tr, groups, pca_d=256):
    """Exact evaluation of the class model (w = 0): per-space PCA + Gaussian
    classifier with pooled covariance, i.e. a shrinkage LDA. The pooled
    ensemble of these readouts is the DeepShip mainline number."""
    yidx = _yidx(y_tr, groups)
    p_ = PCA(pca_d, random_state=0).fit(E_tr)
    X = p_.transform(E_tr)
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X, yidx)

    def predict(E):
        PK = np.zeros((len(E), len(groups)))
        pe = clf.predict_proba(p_.transform(E))
        for j, c in enumerate(clf.classes_):
            PK[:, int(c)] += pe[:, j]
        return PK

    return predict


def rule_w(alpha, lam=LAM):
    """The rule of the paper: w = lam * max(alpha, 0)."""
    return lam * max(alpha, 0.0)


def support_alpha(E_tr, y_tr, Tx, labs, groups, n_template=40):
    """Direction alignment measured on the labeled support: mean cosine
    between audio class-mean differences and text class-mean differences.
    Text class means use the template block (the first n_template rows of
    each class; texts are ordered templates-first). This is the statistic the
    rule reads, computed before any fitting, on the public CLAP-L space."""
    yidx = np.asarray([groups.index(c) for c in y_tr])
    tidx = np.asarray([groups.index(c) for c in labs])
    present = [g for g in range(len(groups)) if (yidx == g).any()]
    amus = np.stack([E_tr[yidx == g].mean(0) for g in present])
    tmus = np.stack([Tx[tidx == g][:n_template].mean(0) for g in present])
    cs = []
    for a in range(len(present)):
        for b in range(a + 1, len(present)):
            da, dt = amus[a] - amus[b], tmus[a] - tmus[b]
            cs.append(float(da @ dt /
                            (np.linalg.norm(da) * np.linalg.norm(dt) + 1e-9)))
    return float(np.mean(cs))


def pool_predict(predicts, E):
    """Ensemble of spaces: equal-weight average of per-space probabilities."""
    return np.mean([p(E) for p in predicts], axis=0)
