# SOURCE: GPLA-12 real acoustic dataset

- **URL**: https://github.com/Deep-AI-Application-DAIP/acoustic-leakage-dataset-GPLA-12
- **commit**: `afec271f4eb445a976826c6a45485892e5fbab53` (shallow clone, depth 1)
- **download date**: 2026-10-11
- **license**: Eclipse Public License 2.0 (EPL-2.0). First lines verbatim:
  1. `Eclipse Public License - v 2.0`
  2. `\n`
  3. `    THE ACCOMPANYING PROGRAM IS PROVIDED UNDER THE TERMS OF THIS ECLIPSE`
- **citation**: J. Li, L. Yao, "GPLA-12: An Acoustic Signal Dataset of Gas Pipeline Leakage", arXiv:2106.10277 (2021), doi:10.48550/arXiv.2106.10277. Note: arXiv preprint license is arXiv nonexclusive-distrib 1.0; the *dataset repository* is EPL-2.0.

## Files (sha256 over repo HEAD `afec271`)
| path | size | sha256 |
|---|---|---|
| data/data_v1/data.csv | 3702331 | 00135BF5F4A417D07219DD9CD340DD3834DD2CBB75AEA7B11FDACFBA2F0FCBA1 |
| data/data_v1/label.csv | 2223 | 97235DD9ABDB0ABF3082BB310C96CA300F276B948E15DACF9465B4EF20997332 |
| data/data_v2/data.xlsx | 3985781 | BF4783E08679F754F7F9DEB076B0F3FF38CCFA02FE2A6DCB383B3F1BD8E2BA2E |
| data/data_v2/lable.xlsx | 14590 | 3EA2342EE6407EF34CB7564BC2D7326D26C2045B0204838BB0E187D46980F00F |
| data/data_v3/data_v3 | 2 | 7EB70257593DA06F682A3DDDA54A9D260D4FC514F645237F5CA74B08F8DA61A6 (placeholder, `\r\n`) |
| LICENSE | 14474 | 6C39067D3BA9D479C7A7256D827D8DE6C6541C786EC77A25C1AC0CDFF8A30E23 |
| README.md | 971 | 2993BEF470028CA734F8AA23D60B73D662264395EF1207CCC403F1691CF77B58 |

## data_v3 note
README states "the newest is v3", but `data_v3/` contains only a 2-byte placeholder file. Real signal data is v1 (CSV, canonical) and v2 (Excel). This project loads **data_v1** only.

## Used subset: data_v1
- **array shapes**: `data.csv` (684, 1460) int64, values 0..1471 (unitless preprocessed amplitudes); `label.csv` (684, 1), classes 1..12.
- **classes**: 12 (leakage categories). **samples per class**: 57 each (684 = 12×57, balanced).
- **sampling rate / units**: not documented in the repo README or the arXiv abstract; signals are "structured tailoring" preprocessed acoustic recordings of 1460 samples each. No fs or units stated.
- **amplitude stats (all samples)**: min=0, max=1471, std=426.444.
- **per-class** (class: min/max/std):
  1: 0/962/146.53, 2: 0/707/81.97, 3: 0/1072/173.92, 4: 0/819/134.59, 5: 0/1052/189.34,
  6: 0/927/157.00, 7: 211/1471/201.14, 8: 456/1271/89.96, 9: 319/1439/155.83,
  10: 496/1352/97.44, 11: 504/1439/128.14, 12: 555/1359/96.65.
- **row activity**: nonzero samples per row mean 1210.2, min 859, max 1460; row sums 119428..1473776.