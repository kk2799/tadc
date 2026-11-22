# Backbones and embedding spaces

TADC consumes frozen feature spaces. Each space is realized as one
`<space>.npz` (audio-block embeddings + labels) plus a text tower for the
library. The unified zero-contact pool is six spaces per dataset: two public
towers, one domain-adapted member, three self-supervised members.

## Public towers (auto-download)

| Space | Checkpoint | Window | Used by |
|-------|-----------|--------|---------|
| `clap_l` | `laion/clap-htsat-fused` (HF) | 10 s | SE, DS, IARA |
| `clap_m` | MS-CLAP 2023 (pip `msclap`) | 10 s | SE, DS, IARA |

`embed_public.py` turns a manifest (CSV: `path,label,split`) into the npz
archives above. On ShipsEar the labels are the nine ship types (fitting is
type-level; the runner maps predictions to the five aggregated classes).

## Domain-adapted and self-supervised members (release archive)

These were trained on the *other two* datasets' audio (leave-one-dataset-out;
no member has seen its pool's target dataset) and ship as frozen embedding
archives (`*.npz`) in the release download:

| Space | Kind | Used by |
|-------|------|---------|
| `da_clap_l` | LAION CLAP audio tower, domain-adapted | SE, DS, IARA |
| `ssl_cqt_1..3` | ResNet18+CQT, self-supervised (three training seeds) | SE, DS, IARA |

Download with `python reproduce.py --download-weights` and place the archives
under `embeddings/<dataset>/`.

## Text towers

Each space embeds the text library with its own tower: CLAP-L spaces (and the
self-supervised members, which have no text side of their own) use the LAION
text tower; the CLAP-M space uses the MS-CLAP 2023 text tower
(`embed_texts.py`).
