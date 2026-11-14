# Backbones and embedding spaces

TADC consumes frozen feature spaces. Each space is realized as one
`<space>.npz` (audio-block embeddings + labels) plus a text tower for the
library. Three groups:

## Public towers (auto-download)

| Space | Checkpoint | Window | Used by |
|-------|-----------|--------|---------|
| `clap_l` | `laion/clap-htsat-fused` (HF) | 10 s | SE, DS, IARA |
| `clap_m` | MS-CLAP 2023 (pip `msclap`) | 10 s | DS, IARA |
| `ast` | `MIT/ast` (HF) | 10 s | SE (via zero-label projection, `tadc/core.py` + Eq. proj) |

`embed_public.py` turns a manifest (CSV: `path,label,split`) into the npz
archives above.

## Domain-adapted and self-supervised members (release archive)

These were trained on dataset audio and ship as frozen embedding archives
(`*.npz`) in the release download:

| Space | Kind | Used by |
|-------|------|---------|
| `da_clap_l` | LAION CLAP audio tower, domain-adapted | SE, DS, IARA |
| `da_clap_m_a`, `da_clap_m_b` | MS-CLAP audio tower, two runs | SE |
| `da_htsat`, `da_beats` | HTSAT / BEATs, domain-adapted | SE |
| `ssl_cqt_1..3` | ResNet18+CQT, self-supervised on DeepShip audio | DS |

Download with `python reproduce.py --download-weights` and place the archives
under `embeddings/<dataset>/`.

## Text towers

Each space embeds the text library with its own tower: CLAP-L spaces use the
LAION text tower, CLAP-M spaces use the MS-CLAP 2023 text tower, AST-derived
spaces use the LAION text tower (the projection of Eq. (proj) aligns the
audio side to it; no labels involved).
