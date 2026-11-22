# TADC: Text-Anchored Distribution Calibration

Label-efficient underwater acoustic target recognition (UATR) on frozen audio
encoders. TADC synthesizes a training set in each frozen feature space — the
class mean gives the location, the centered text anchor gives the direction,
the pooled within-class covariance gives the shape — and fits a linear readout
on synthetic samples only. Text is used at fit time; deployment is pure audio.

> **Paper**: *Text-Anchored Distribution Calibration for Underwater Acoustic
> Target Recognition* (under review).

## Results

Unified zero-contact pool: six frozen spaces per dataset (two public CLAP
encoders, one domain-adapted encoder, three self-supervised encoders), none of
which has seen the target dataset's audio. Mean ± std over three synthesis
seeds; IARA is the official 5×2 cross-validation with recording-level voting:

| Dataset  | Metric        | TADC           | Same-pool full-label probe |
|----------|---------------|----------------|---------------------------|
| ShipsEar | window acc    | **85.92 ± 0.41** | 81.03 |
| DeepShip | window acc    | **78.25** (closed form) | 77.82 |
| IARA     | balanced acc  | **70.03 ± 0.01** | 70.84 |

The injection weight follows one rule on all three datasets:
`w = λ·max(α, 0)` with `λ = 0.40`, where α is the direction alignment between
audio class-mean differences and text class-mean differences, measured on the
labeled support before any fitting. α is positive on ShipsEar (+0.10, w ≈ 0.04)
and negative on DeepShip (−0.01) and IARA (−0.02), where the rule sets w = 0
and the readout reduces to the exact evaluation of the class model.

## Install

```bash
pip install -r requirements.txt
```

Encoders are pulled automatically on first use (LAION CLAP via `transformers`,
MS-CLAP 2023 via `msclap`). The domain-adapted and self-supervised spaces ship
as frozen embedding archives (`python reproduce.py --download-weights`).

## Repository layout

```
tadc/core.py        synthesis + readout + rule + closed form
runners/            per-dataset mainline runners (unified six-space pool)
backbones/          embedding / packing utilities
data/text/          text libraries: 40 template repeats + 40 generated
                    sentences per class (80 rows per class, exactly the
                    libraries used in the paper)
splits/             per-file train/test splits
embeddings/         per-space npz archives (built by the steps below)
```

## Quickstart (DeepShip)

1. Download DeepShip and place the `.wav` files by their dataset IDs; the
   train/test split is `splits/deepship_test_ids.csv` (files not listed are
   training).
2. Cut 30-s blocks, embed them with each pool space, and save one
   `<space>.npz` per space under `embeddings/deepship/` with arrays
   `E_tr, y_tr, E_te, y_te` (`backbones/embed_public.py` for the public
   towers, `python reproduce.py --download-weights` for the rest), then
   attach the text side (`backbones/embed_texts.py`).
3. Run:

```bash
python runners/run_deepship.py --emb-dir embeddings/deepship \
    --spaces clap_l clap_m da_clap_l ssl_cqt_1 ssl_cqt_2 ssl_cqt_3p
```

On DeepShip the rule reads α = −0.01 and sets w = 0, so the runner reports
the closed-form evaluation of the class model: **78.25**, the paper's number.
`--sampled` draws from the same class model instead (~77.7). ShipsEar and
IARA follow the same pattern (`runners/`); IARA uses the official 5×2
cross-validation with recording-level voting.

## Text libraries

`data/text/<dataset>.json` is the single source of truth — exactly the
library used in the paper. Per class: one template sentence (fixed frame:
class name, propulsion, perceived timbre) repeated 40 times as a
deterministic anchor kernel, plus 40 generated sentences reviewed by hand.
Rows are ordered templates-first, so the rule's text class means are the
first 40 rows of each class. On IARA the gate is closed and only the
template rows enter the alignment measurement; no text row enters the fit.

| file | structure |
|------|-----------|
| `shipsear.json` | `templates` (1 per type) + `llm` (40 per type) |
| `deepship.json` | `templates` (1 per class: cargo / passenger / tanker / tug) + `llm` (40 per class) |
| `iara.json` | `templates` (1 per class) + `llm` (40 per class) |

## Splits

- `splits/shipsear_train_test.csv` — per-file train/test IDs (ship-disjoint).
- `splits/deepship_test_ids.csv` — per-category test file IDs (train = the
  rest; transductive setting, the same ships may appear on both sides).
- `splits/iara_recordings.csv` — recording IDs and length grades for the
  official 5×2 cross-validation.

## Reproduce everything

```bash
python reproduce.py --download-weights   # frozen space archives
python reproduce.py --dataset all
```

Expected: ShipsEar 85.92 ± 0.41, DeepShip 78.25 (closed form), IARA
70.03 ± 0.01.

## Citation

```bibtex
@article{tadc2026,
  title   = {Text-Anchored Distribution Calibration for Underwater Acoustic
             Target Recognition},
  year    = {2026},
  note    = {under review}
}
```

The DeepShip split protocol follows Liu et al., *JASA* 160:1182–1198 (2026),
doi:10.1121/10.0044109.

## License

MIT (see LICENSE).
