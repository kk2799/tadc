# -*- coding: utf-8 -*-
"""Embed audio blocks with the public towers used by TADC.

Covers the two publicly downloadable spaces: CLAP-L (LAION
clap-htsat-fused, 10-s windows at 48 kHz) and CLAP-M (MS-CLAP 2023).
The domain-adapted and self-supervised members (DA-CLAP, SSL-CQT instances)
ship as frozen embedding archives in the release; see backbones/README.md.

Usage:
  python backbones/embed_public.py --dataset deepship \
      --manifest manifests/deepship.csv --out embeddings/deepship
The manifest lists one audio block per row: path,label,split
(split in {train, test}). One <tower>.npz is written per tower with
E_tr, y_tr, E_te, y_te.
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly

TOWERS = {
    "clap_l": "laion/clap-htsat-fused",
    # "clap_m" uses the msclap package (version="2023"); enabled below.
}


def load_audio(p, target_sr=48000, max_sec=10):
    y, sr = sf.read(str(p), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(1)
    if sr != target_sr:
        y = resample_poly(y, target_sr, sr).astype("float32")
    cap = target_sr * max_sec
    if len(y) > cap:
        s = (len(y) - cap) // 2
        y = y[s:s + cap]
    return np.pad(y, (0, max(0, cap - len(y))))


def embed_clap_l(paths, device="cpu", batch=8):
    from transformers import ClapModel, ClapProcessor
    model = ClapModel.from_pretrained(TOWERS["clap_l"], dtype=torch.float32).to(device).eval()
    proc = ClapProcessor.from_pretrained(TOWERS["clap_l"])
    out = []
    with torch.no_grad():
        for s0 in range(0, len(paths), batch):
            wavs = [load_audio(p) for p in paths[s0:s0 + batch]]
            inp = proc(audio=wavs, sampling_rate=48000, return_tensors="pt", padding=True)
            inp = {k: v.to(device) for k, v in inp.items() if k in ("input_features", "is_longer")}
            out.append(F.normalize(model.get_audio_features(**inp).float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


def embed_clap_m(paths, batch=64):
    from msclap import CLAP
    cl = CLAP(version="2023", use_cuda=False)
    out = []
    with torch.no_grad():
        for s0 in range(0, len(paths), batch):
            wavs = [load_audio(p) for p in paths[s0:s0 + batch]]
            out.append(F.normalize(cl.get_audio_embeddings(wavs).float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


EMBEDDERS = {"clap_l": embed_clap_l, "clap_m": embed_clap_m}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="CSV: path,label,split")
    ap.add_argument("--out", required=True)
    ap.add_argument("--towers", nargs="+", default=["clap_l", "clap_m"])
    args = ap.parse_args()

    rows = [l.strip().split(",") for l in open(args.manifest, encoding="utf-8") if l.strip()]
    paths = [r[0] for r in rows]
    labels = np.array([r[1] for r in rows])
    is_te = np.array([r[2] == "test" for r in rows])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for t in args.towers:
        E = EMBEDDERS[t](paths)
        np.savez(out / f"{t}.npz",
                 E_tr=E[~is_te], y_tr=labels[~is_te],
                 E_te=E[is_te], y_te=labels[is_te])
        print(f"{t}: {len(E)} blocks embedded -> {out / (t + '.npz')}")


if __name__ == "__main__":
    main()
