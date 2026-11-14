# TADC: Text-Anchored Distribution Calibration

Label-efficient underwater acoustic target recognition (UATR) on frozen audio
encoders. TADC synthesizes a training set in each frozen feature space — the
class base gives the location, the centered text anchor gives the direction,
the pooled within-class covariance gives the shape — and fits a linear readout
on synthetic samples only. Text is used at fit time; deployment is pure audio.

> **Paper**: *Label-efficient underwater acoustic target recognition via
> text-anchored distribution calibration* (under review).

## Results

Main configuration (greedy-selected space pool per dataset, mean ± std over
5 synthesis seeds; IARA is the official 5×2 cross-validation protocol with
recording-level voting):

| Dataset  | Metric        | TADC (paper)  | TADC (this repo) | Full-label logistic regression |
|----------|---------------|---------------|------------------|-------------------------------|
| ShipsEar | window acc    | 86.38 ± 1.01  | 88.62 ± 0.34     | 81.38 ± 0.42 |
| DeepShip | window acc    | 77.50 ± 0.79  | 77.35 ± 0.76     | 75.07 ± 0.46 |
| IARA     | balanced acc  | 68.43 ± 1.30  | 68.21 ± 0.96     | 69.01 ± 0.93 |

DeepShip and IARA reproduce within noise. ShipsEar exceeds the paper by
2.24 points because the freshly computed PROJ-AST projection (seed 0)
happens to align better than the original — the paper's number is the
conservative one.

On ShipsEar and DeepShip, **10% of the labels already reaches full-annotation
accuracy** (82.93 and 75.21).

## Install

```bash
pip install -r requirements.txt
```

Encoders are pulled automatically on first use (LAION CLAP via `transformers`,
MS-CLAP 2023 via `msclap`). No domain training of any backbone is required
for the CLAP-family spaces.

## Repository layout

```
tadc/core.py        synthesis + readout + gate + pooling (Eq. 1-2 of the paper)
runners/            per-dataset mainline runners
data/text/          text libraries: 40 template sentences per class + curated
                    LLM-written descriptions (the paper's full libraries)
splits/             per-file train/test splits (CSV)
```

## Quickstart (DeepShip)

1. Download DeepShip and place the `.wav` files by their dataset IDs; the
   train/test split is `splits/deepship_test_ids.csv` (files not listed are
   training; the split follows the protocol of the paper's §4.1).
2. Cut 30-s blocks, embed them with each pool space (CLAP-L / CLAP-M /
   DA-CLAP-L towers + the SSL-CQT instances), and save one
   `<space>.npz` per space under `embeddings/deepship/` with arrays
   `E_tr, y_tr, E_te, y_te, Tx, labs`.
3. Run:

```bash
python runners/run_deepship.py --emb-dir embeddings/deepship \
    --spaces clap_l clap_m da_clap_l ssl_cqt_1 ssl_cqt_2 ssl_cqt_3p
```

The gate (direction alignment, computed before any fitting) sets the
injection weight: `w = 0.02` where alignment is positive, `w = 0` otherwise —
on DeepShip the gate selects `w = 0`, i.e. the pure class-statistics route.

ShipsEar and IARA follow the same pattern (`runners/`); the ShipsEar pool
additionally uses domain-adapted towers and a zero-label-projected AST space
(checkpoints and extraction scripts are released alongside on acceptance),
and IARA uses the official 5×2 cross-validation with recording-level voting.

## Text libraries

`data/text/*.json` holds the LLM-written descriptions used in the paper
(curated and human-reviewed sentence by sentence); each space additionally
uses 40 template sentences per class (fixed frame: class name, propulsion,
perceived timbre).

| file | structure |
|------|-----------|
| `shipsear.json` | `templates` (1 per type) + `llm` (198 per type, ordered) |
| `deepship.json` | `templates` (1 per class, named: cargo ship / passenger ship / oil tanker / tug) + `llm` (198 per class, ordered) |
| `iara.json` | `templates` (1 per class) + `llm` (412 per class, ordered) |

At fit time each template is repeated 40 times per class, serving as a
deterministic anchor kernel while the generated sentences spread the
within-class diversity. The library is consumed by prefix: the mainline
results use the first 40 LLM sentences per type/class for ShipsEar and
DeepShip and the first 200 per class for IARA (matching the paper's §4.1
totals of 360, 160, and 200), and the budget sweeps use the full list. The
gate makes TADC robust to the library size: two sentences per class already
saturate ShipsEar accuracy.

## Splits

- `splits/shipsear_train_test.csv` — per-file train/test IDs (ship-disjoint;
  the split of the paper's §4.1).
- `splits/deepship_test_ids.csv` — per-category test file IDs (train = the
  rest; transductive setting, the same ships may appear on both sides).

## Citation

```bibtex
@article{tadc2026,
  title   = {Label-efficient underwater acoustic target recognition via
             text-anchored distribution calibration},
  year    = {2026},
  note    = {under review}
}
```

The DeepShip split protocol follows Liu et al., *JASA* 160:1182–1198 (2026),
doi:10.1121/10.0044109.

## License

MIT (see LICENSE).
