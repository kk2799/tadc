# -*- coding: utf-8 -*-
"""One-click reproduction of the TADC mainline results.

  python reproduce.py --dataset all            # run everything present
  python reproduce.py --dataset deepship       # one dataset
  python reproduce.py --download-weights       # fetch frozen space archives

Paper numbers (unified zero-contact pool, six spaces per dataset, three
synthesis seeds; IARA is the official 5x2 cross-validation with recording
voting; DeepShip is the closed-form evaluation, exact):

  ShipsEar  85.92 +/- 0.41
  DeepShip  78.25
  IARA      70.03 +/- 0.01

Per-dataset runners live in runners/; each consumes per-space npz archives
under embeddings/<dataset>/ (built by backbones/embed_public.py for public
towers, downloaded for adapted/self-supervised ones) and the text libraries
in data/text/ (attached by backbones/embed_texts.py).
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

SPACES = ["clap_l", "clap_m", "da_clap_l", "ssl_cqt_1", "ssl_cqt_2", "ssl_cqt_3p"]
PAPER = {
    "shipsear": ("85.92 +/- 0.41", SPACES),
    "deepship": ("78.25 (closed form)", SPACES),
    "iara": ("70.03 +/- 0.01", SPACES),
}
WEIGHTS_TAG = "v2.0"  # release tag carrying tadc_spaces.zip (frozen space archives)


def _weights_url():
    """Derive the release asset URL from the git remote at run time, so no
    repository owner is hard-coded in this file."""
    import re as _re
    import subprocess
    try:
        remote = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        remote = ""
    m = _re.search(r"github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$", remote)
    if not m:
        raise SystemExit(
            "cannot locate the release: download tadc_spaces.zip from the "
            "repository's Releases page (tag " + WEIGHTS_TAG + ") and unpack "
            "it into embeddings/"
        )
    owner, repo = m.group(1), m.group(2)
    return f"https://github.com/{owner}/{repo}/releases/download/{WEIGHTS_TAG}/tadc_spaces.zip"


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
    url = _weights_url()
    print("downloading", url)
    dest = ROOT / "embeddings"
    dest.mkdir(exist_ok=True)
    z, _ = urllib.request.urlretrieve(url, dest / "tadc_spaces.zip")
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
