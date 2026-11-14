# -*- coding: utf-8 -*-
"""Embed the text libraries with each space's text tower and attach Tx/labs
to the space npz archives.

Towers: clap_l (LAION text tower, used by CLAP-L / DA-CLAP-L / AST-projected
spaces), clap_m (MS-CLAP 2023 text tower, used by CLAP-M spaces).

Usage:
  python backbones/embed_texts.py --dataset deepship \
      --emb-dir embeddings/deepship \
      --spaces clap_l:clap_l clap_m:clap_m da_clap_l:clap_l ssl_cqt_1:clap_l ...

Each `space:tower` pair means: append Tx, labs (embedded with that text
tower) to embeddings/<dataset>/<space>.npz. The library is the mainline
prefix: 40 template repeats + first-40 (SE/DS) or first-200 (IARA) LLM
sentences per class, per the paper §4.1.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]

TEMPLATES = {
    "shipsear": {
        "Dredger": ("diesel engines driving suction pumps", "a continuous pumping machinery noise"),
        "Fishboat": ("a small auxiliary diesel engine", "a light irregular diesel clatter"),
        "Motorboat": ("a high-revving outboard engine", "a sharp high-pitched engine whine"),
        "Musselboat": ("a small diesel with dredge machinery", "a rhythmic machinery rattle"),
        "Naturalambientnoise": ("no engine, only wind and waves", "natural sea noise"),
        "Oceanliner": ("a large slow-speed diesel engine", "a deep steady hum"),
        "Passengers": ("medium-speed twin diesels", "a regular medium-tone rhythm"),
        "RORO": ("a large slow-revving diesel", "a deep continuous machinery tone"),
        "Sailboat": ("no engine, sails only", "quiet water flow"),
    },
    "deepship": {
        "0": ("a slow-speed diesel engine with a fixed-pitch propeller", "a deep steady drone"),
        "1": ("medium-speed diesel propulsion", "a regular medium-tone rhythm"),
        "2": ("a slow-speed diesel engine, deeply laden", "a heavy low-frequency rumble"),
        "3": ("azimuth thrusters driven by high-rate diesels", "a high-rate churning thrash"),
    },
    "iara": {
        "SMALL": ("a compact high-revving engine", "a light high-pitched engine whine"),
        "MEDIUM": ("a medium-speed diesel engine", "a regular medium-tone rhythm"),
        "LARGE": ("a large slow-speed diesel engine", "a heavy low-frequency rumble"),
        "BG": ("no ship nearby", "quiet sea"),
    },
}
NAME = {
    "shipsear": lambda c: c.lower(),
    "deepship": {"0": "cargo ship", "1": "passenger ship", "2": "oil tanker", "3": "tug"},
    "iara": {"SMALL": "small vessel under 50 m", "MEDIUM": "medium vessel of 50 to 100 m",
             "LARGE": "large ship over 100 m", "BG": "quiet sea"},
}
MAINLINE_LLM = {"shipsear": 40, "deepship": 40, "iara": 200}
TOWER_MODEL = {"clap_l": "laion/clap-htsat-fused"}


def se_frame(name, a, b):
    art = "an" if name[0] in "aeiou" else "a"
    return f"The underwater sound of {art} {name} with {a}, heard as {b}."


def ds_frame(c, a, b):
    return f"The underwater sound of a {NAME['deepship'][c]} with {a}, heard as {b}."


def ia_frame(c, a, b):
    return f"The underwater sound of a {NAME['iara'][c]}, with {a}, heard as {b}."


def build_mainline(dataset):
    lib = json.load(open(ROOT / "data" / "text" / f"{dataset}.json", encoding="utf-8"))
    texts, labs = [], []
    tmpl = TEMPLATES[dataset]
    for c in tmpl:
        if dataset == "shipsear":
            base = se_frame(NAME[dataset](c), *tmpl[c])
        elif dataset == "deepship":
            base = ds_frame(c, *tmpl[c])
        else:
            base = ia_frame(c, *tmpl[c])
        texts += [base] * 40
        labs += [c] * 40
        for s in lib["llm"][c][:MAINLINE_LLM[dataset]]:
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
    ap.add_argument("--dataset", required=True, choices=list(TEMPLATES))
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--spaces", nargs="+", required=True, help="space:tower pairs")
    args = ap.parse_args()

    texts, labs = build_mainline(args.dataset)
    print(f"mainline library: {len(texts)} sentences")
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
