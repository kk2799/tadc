# -*- coding: utf-8 -*-
"""Collect the domain-adapted / self-supervised space archives from the
experiment tree into the release layout (embeddings/ + dist/tadc_spaces.zip).

Unified zero-contact pool, six spaces per dataset; the public towers
(clap_l, clap_m) are NOT packed -- they are rebuilt by embed_public.py from
the public checkpoints, and their text sides are attached by embed_texts.py.

Layout produced:
  embeddings/shipsear/{da_clap_l, ssl_cqt_1, ssl_cqt_2, ssl_cqt_3p}.npz
      y_tr/y_te carry the nine ship-type labels (fitting is type-level; the
      runner maps predictions to the five aggregated classes A-E).
  embeddings/deepship/{da_clap_l, ssl_cqt_1, ssl_cqt_2, ssl_cqt_3p}.npz
  embeddings/iara/{da_clap_l, ssl_cqt_1, ssl_cqt_2, ssl_cqt_3p}.npz
      IARA spaces carry E/y/iid/j plus m (window-validity mask).

The script also re-labels the public ShipsEar npz (clap_l/clap_m) to the
nine-type level and removes the withdrawn greedy-pool archives
(da_clap_m_a/b, da_htsat, da_beats, proj_ast).
"""
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REL = HERE.parents[0]                    # code_release/
SRC = HERE.parents[2] / "af3_experiment" / "results"
DATA = HERE.parents[2] / "af3_experiment" / "data_uart"
SSL_SE = {"ssl_cqt_1": "se_ssl_cross_seg.npz",
          "ssl_cqt_2": "se_ssl_cross_seg_s43.npz",
          "ssl_cqt_3p": "se_ssl_cross_seg_s45.npz"}
SSL_IA = {"ssl_cqt_1": "cross_ssl_iara_s42.npz",
          "ssl_cqt_2": "cross_ssl_iara_s43.npz",
          "ssl_cqt_3p": "cross_ssl_iara_s45.npz"}


def npz_bytes(arr):
    import io
    buf = io.BytesIO()
    np.savez(buf, **arr)
    return buf.getvalue()


def se_meta():
    df = pd.read_csv(DATA / "uart_meta_v2.csv").reset_index(drop=True)
    tr = (df.split == "train").values
    return df, tr


def pack_shipsear(z):
    df, tr = se_meta()
    y_tr = df.type.values[tr].astype(str)
    y_te = df.type.values[~tr].astype(str)

    # da_clap_l: LAION audio tower, domain-adapted (experiment source:
    # crossda_se_emb.npz, 601 blocks with its own split column)
    zd = np.load(SRC / "crossda_se_emb.npz", allow_pickle=True)
    m_da = zd["split"].astype(str) == "train"
    z.writestr("embeddings/shipsear/da_clap_l.npz",
               npz_bytes(dict(E_tr=zd["E"][m_da], y_tr=y_tr,
                              E_te=zd["E"][~m_da], y_te=y_te)))

    # ssl_cqt instances: normalized, split from their own split column
    for rel, fn in SSL_SE.items():
        zd = np.load(SRC / fn, allow_pickle=True)
        E = zd["E"] / (np.linalg.norm(zd["E"], axis=1, keepdims=True) + 1e-9)
        m = zd["split"].astype(str) == "train"
        z.writestr(f"embeddings/shipsear/{rel}.npz",
                   npz_bytes(dict(E_tr=E[m], y_tr=y_tr, E_te=E[~m], y_te=y_te)))

    # re-label the public towers to the nine-type level (fitting is type-level)
    for pub in ["clap_l", "clap_m"]:
        fp = REL / "embeddings" / "shipsear" / f"{pub}.npz"
        if fp.exists():
            d = dict(np.load(fp, allow_pickle=True))
            if len(set(d["y_tr"].tolist())) == 5:
                d["y_tr"], d["y_te"] = y_tr, y_te
                np.savez(fp, **d)
                print(f"re-labelled {pub} to nine-type labels")

    # withdraw the greedy-pool archives no longer in the unified pool
    for stale in ["da_clap_m_a", "da_clap_m_b", "da_htsat", "da_beats", "proj_ast"]:
        fp = REL / "embeddings" / "shipsear" / f"{stale}.npz"
        if fp.exists():
            fp.unlink()
            print(f"removed stale {stale}.npz")


def pack_deepship(z):
    # experiment sources: ds_zc() in _zc_members.py
    zc = np.load(SRC / "crossda_ds_refemb.npz", allow_pickle=True)
    z.writestr("embeddings/deepship/da_clap_l.npz",
               npz_bytes(dict(E_tr=zc["E_tr"], y_tr=zc["y_tr"].astype(str),
                              E_te=zc["E_te"], y_te=zc["y_te"].astype(str))))
    for t, rel in [("s42", "ssl_cqt_1"), ("s43", "ssl_cqt_2"), ("s45", "ssl_cqt_3p")]:
        zx = np.load(SRC / f"cross_ssl_ds_{t}.npz", allow_pickle=True)
        z.writestr(f"embeddings/deepship/{rel}.npz",
                   npz_bytes(dict(E_tr=zx["E_tr"], y_tr=zx["y_tr"].astype(str),
                                  E_te=zx["E_te"], y_te=zx["y_te"].astype(str))))


def pack_iara(z):
    zl = np.load(SRC / "iara_offcpa_laion.npz", allow_pickle=True)
    iid = zl["iid"].astype(int)
    jj = zl["j"]
    y = zl["y"].astype(str)
    n = len(iid)
    # experiment source for the domain-adapted space: crossda_iara_emb.npz
    zia = np.load(SRC / "crossda_iara_emb.npz", allow_pickle=True)
    assert np.array_equal(zia["iid"].astype(int), iid)
    z.writestr("embeddings/iara/da_clap_l.npz",
               npz_bytes(dict(E=zia["E"], y=y, iid=iid)))
    key2row = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(iid, jj))}
    for rel, fn in SSL_IA.items():
        E = np.full((n, 512), np.nan)
        with np.load(SRC / fn, allow_pickle=True) as zd:
            for fi, name in enumerate(zd["names"].tolist()):
                p = name[:-4].split("_")
                r = key2row.get((int(p[0]), int(p[1])))
                if r is not None:
                    E[r] = zd["E"][fi]
        m = ~np.isnan(E[:, 0])
        z.writestr(f"embeddings/iara/{rel}.npz",
                   npz_bytes(dict(E=E, y=y, iid=iid, m=m)))


def main():
    (REL / "dist").mkdir(exist_ok=True)
    out = REL / "dist" / "tadc_spaces.zip"
    z = zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED)
    pack_shipsear(z)
    pack_deepship(z)
    pack_iara(z)
    z.close()
    # also lay the archives out under embeddings/ for local verification
    with zipfile.ZipFile(out) as zf:
        zf.extractall(REL)
    print(f"packed -> {out} ({len(zfile_names(out))} entries); "
          f"extracted into embeddings/")


def zfile_names(fp):
    with zipfile.ZipFile(fp) as zf:
        return zf.namelist()


if __name__ == "__main__":
    main()
