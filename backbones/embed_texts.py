# -*- coding: utf-8 -*-
"""Embed the text libraries with each space's text tower and attach Tx/labs
to the space npz archives.

Towers: clap_l (LAION text tower, used by CLAP-L / DA-CLAP-L / SSL spaces),
clap_m (MS-CLAP 2023 text tower, used by the CLAP-M space).

The library comes from data/text/<dataset>.json (the single source of truth,
exactly the library used in the paper): per class, the template sentence is
repeated 40 times first (the deterministic anchor kernel), then the 40
generated sentences follow; 80 rows per class in total. The same ordering
lets tadc.core.support_alpha take the template block as each class's first
40 rows.

Usage:
  python backbones/embed_texts.py --dataset deepship \
      --emb-dir embeddings/deepship \
      --spaces clap_l:clap_l clap_m:clap_m da_clap_l:clap_l ssl_cqt_1:clap_l ...
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
TOWER_MODEL = {"clap_l": "laion/clap-htsat-fused"}


def build_library(dataset):
    lib = json.load(open(ROOT / "data" / "text" / f"{dataset}.json",
                         encoding="utf-8"))
    texts, labs = [], []
    for c in lib["templates"]:
        texts += [lib["templates"][c]] * 40
        labs += [c] * 40
        for s in lib["llm"][c]:
            texts.append(s)
            labs.append(c)
    return texts, np.array(labs)


def embed_l(texts, batch=48):
    from transformers import ClapModel, ClapProcessor
    mp = TOWER_MODEL["clap_l"]
    proc = ClapProcessor.from_pretrained(mp)
    model = ClapModel.from_pretrained(mp, dtype=torch.float32).eval()
    out = []
    with torch.no_grad():
        for s0 in range(0, len(texts), batch):
            inp = proc(text=texts[s0:s0 + batch], return_tensors="pt", padding=True)
            out.append(F.normalize(model.get_text_features(
                input_ids=inp["input_ids"], attention_mask=inp["attention_mask"]).float(),
                dim=-1).cpu().numpy())
    return np.concatenate(out)


def embed_m(texts, batch=64):
    from msclap import CLAP
    cl = CLAP(version="2023", use_cuda=False)
    cl.clap.eval()
    out = []
    with torch.no_grad():
        for s0 in range(0, len(texts), batch):
            out.append(F.normalize(cl.get_text_embeddings(texts[s0:s0 + batch]).float(),
                                   dim=-1).cpu().numpy())
    return np.concatenate(out)


EMBED = {"clap_l": embed_l, "clap_m": embed_m}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["shipsear", "deepship", "iara"])
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--spaces", nargs="+", required=True, help="space:tower pairs")
    args = ap.parse_args()

    texts, labs = build_library(args.dataset)
    print(f"library: {len(texts)} sentences, classes {sorted(set(labs.tolist()))}")
    cache = {}
    for pair in args.spaces:
        space, tower = pair.split(":")
        if tower not in cache:
            cache[tower] = EMBED[tower](texts)
        fp = Path(args.emb_dir) / f"{space}.npz"
        z = dict(np.load(fp, allow_pickle=True))
        z["Tx"], z["labs"] = cache[tower], labs
        np.savez(fp, **z)
        print(f"{space}: Tx {cache[tower].shape} -> {fp}")


if __name__ == "__main__":
    main()
