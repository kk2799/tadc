# -*- coding: utf-8 -*-
"""One-click reproduction of the TADC mainline results.

  python reproduce.py --dataset all            # run everything present
  python reproduce.py --dataset deepship       # one dataset
  python reproduce.py --download-weights       # fetch frozen space archives

Paper numbers (mean +/- std over 5 synthesis seeds; IARA is the official
5x2 cross-validation with recording voting):

  ShipsEar  86.38 +/- 1.01   (7 spaces)
  DeepShip  77.50 +/- 0.79   (6 spaces)
  IARA      68.43 +/- 1.30   (3 spaces)

Per-dataset runners live in runners/; each consumes per-space npz archives
under embeddings/<dataset>/ (built by backbones/embed_public.py for public
towers, downloaded for adapted/self-supervised ones) and the text libraries
in data/text/.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

PAPER = {
    "shipsear": ("86.38 +/- 1.01", ["clap_l", "da_clap_l", "da_clap_m_a",
                                    "da_clap_m_b", "da_htsat", "da_beats", "proj_ast"]),
    "deepship": ("77.50 +/- 0.79", ["clap_l", "clap_m", "da_clap_l",
                                    "ssl_cqt_1", "ssl_cqt_2", "ssl_cqt_3p"]),
    "iara": ("68.43 +/- 1.30", ["clap_l", "clap_m", "da_clap_l"]),
}
WEIGHTS_URL = "https://github.com/kk2799/tadc/releases/download/v1.0/tadc_spaces.zip"


def have_embeddings(ds, spaces):
    d = ROOT / "embeddings" / ds
    return [s for s in spaces if not (d / f"{s}.npz").exists()]


def run(ds):
    paper_val, spaces = PAPER[ds]
    missing = have_embeddings(ds, spaces)
    if missing:
        print(f"[{ds}] missing embeddings for: {missing}")
        print(f"[{ds}] build them with backbones/embed_public.py (public towers)")
        print(f"[{ds}] and `python reproduce.py --download-weights` (adapted/self-supervised)")
        return None
    r = subprocess.run([sys.executable, str(ROOT / "runners" / f"run_{ds}.py"),
                        "--emb-dir", str(ROOT / "embeddings" / ds),
                        "--spaces", *spaces], capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr)
        return None
    return r.stdout.strip().splitlines()[-1]


def download_weights():
    import urllib.request
    import zipfile
    print("downloading", WEIGHTS_URL)
    dest = ROOT / "embeddings"
    dest.mkdir(exist_ok=True)
    z, _ = urllib.request.urlretrieve(WEIGHTS_URL, dest / "tadc_spaces.zip")
    with zipfile.ZipFile(z) as f:
        f.extractall(dest)
    print("extracted to", dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all",
                    choices=["all", "shipsear", "deepship", "iara"])
    ap.add_argument("--download-weights", action="store_true")
    args = ap.parse_args()

    if args.download_weights:
        download_weights()
        return
    todo = list(PAPER) if args.dataset == "all" else [args.dataset]
    for ds in todo:
        print(f"===== {ds} (paper: {PAPER[ds][0]}) =====")
        got = run(ds)
        if got:
            print(f"[{ds}] reproduced: {got}   paper: {PAPER[ds][0]}")


if __name__ == "__main__":
    main()
