# -*- coding: utf-8 -*-
"""Collect the domain-adapted / self-supervised space archives from the
experiment tree into the release download layout (tadc_spaces.zip).

Layout produced (mirrors what `reproduce.py --download-weights` extracts):
  embeddings/shipsear/{da_clap_l,da_clap_m_a,da_clap_m_b,da_htsat,da_beats}.npz
  embeddings/deepship/{da_clap_l,ssl_cqt_1,ssl_cqt_2,ssl_cqt_3p}.npz
  embeddings/iara/{da_clap_l}.npz

Each npz carries the audio-block embeddings plus labels (E/y_tr, E/y_te or
E/y/iid per protocol). The public towers (clap_l, clap_m, ast) are NOT
packed — they are rebuilt by embed_public.py from the public checkpoints.
Text sides (Tx/labs) are attached at the user's machine by embed_texts.py.
"""
import zipfile
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[2] / "af3_experiment" / "results"  # relative to paper_submission_v3/code_release/backbones/
SSL = SRC.parent.parent / "SPLASH" / "outputs"
OUT = Path(__file__).resolve().parents[1] / "dist"
MAP = {
    # release name -> (source npz, protocol-specific key remap)
    ("shipsear", "da_clap_l"): None,  # SE 域适应塔：见下方实际来源映射
    ("deepship", "da_clap_l"): (SRC / "ds_theirsplit_emb.npz", "ED"),
    ("iara", "da_clap_l"): (SRC / "iara_offcpa_da.npz", None),
}


def main():
    OUT.mkdir(exist_ok=True)
    z = zipfile.ZipFile(OUT / "tadc_spaces.zip", "w", zipfile.ZIP_DEFLATED)
    # DS: da_clap_l = ED 字段（通用域适应塔）
    src = np.load(SRC / "ds_theirsplit_emb.npz", allow_pickle=True)
    n_tr = int(src["n_tr"])
    y = src["y"].astype(str)
    arr = dict(E_tr=src["ED"][:n_tr], y_tr=y[:n_tr], E_te=src["ED"][n_tr:], y_te=y[n_tr:])
    z.writestr("embeddings/deepship/da_clap_l.npz", _npz_bytes(arr))
    # DS: ssl_cqt_1..3 = d1b / d1bs43 / d1bs45p（贪心链入选的三实例）
    for rel, fp in [("ssl_cqt_1", SSL / "d1b_selfsup_cqt_ds.npz"),
                    ("ssl_cqt_2", SSL / "d1b_s43.npz"),
                    ("ssl_cqt_3p", SSL / "d1b_s45.npz")]:
        zd = np.load(fp, allow_pickle=True)
        arr = dict(E_tr=zd["E_tr"], y_tr=zd["y_tr"].astype(str),
                   E_te=zd["E_te"], y_te=zd["y_te"].astype(str))
        z.writestr(f"embeddings/deepship/{rel}.npz", _npz_bytes(arr))
    # IARA: da_clap_l（官方协议嵌入，含 iid）
    zi = np.load(SRC / "iara_offcpa_da.npz", allow_pickle=True)
    arr = dict(E=zi["E"], y=zi["y"].astype(str), iid=zi["iid"].astype(int))
    z.writestr("embeddings/iara/da_clap_l.npz", _npz_bytes(arr))

    # SE: 域适应/自监督塔（spaces56.npz: 485 train / 116 test 段级嵌入，1024 维）
    z56 = np.load(SRC / "spaces56.npz", allow_pickle=True)
    lab = np.load(Path(__file__).resolve().parent / "_se_labels.npz", allow_pickle=True)
    yl_tr = lab["y_tr"].astype(str)
    yl_te = lab["y_te"].astype(str)

    se_map = {"da_clap_m_a": "da35pure", "da_clap_m_b": "da35text",
              "da_htsat": "htsat_ta", "da_beats": "beats"}
    for rel, key in se_map.items():
        arr = dict(E_tr=z56[f"{key}_tr"], y_tr=yl_tr,
                   E_te=z56[f"{key}_te"], y_te=yl_te)
        z.writestr(f"embeddings/shipsear/{rel}.npz", _npz_bytes(arr))

    # SE: da_clap_l（LAION 音频塔域适应，601 段 = 485 train + 116 test）
    da = np.load(SRC.parent / "data_uart" / "laion_da_emb.npz", allow_pickle=True)
    A = da["A"]
    arr = dict(E_tr=A[:485], y_tr=yl_tr, E_te=A[485:], y_te=yl_te)
    z.writestr("embeddings/shipsear/da_clap_l.npz", _npz_bytes(arr))

    z.close()
    print("packed ->", OUT / "tadc_spaces.zip")


def _npz_bytes(arr):
    import io
    buf = io.BytesIO()
    np.savez(buf, **arr)
    return buf.getvalue()


if __name__ == "__main__":
    main()
