# Troubleshooting

| Symptom | Likely cause / remedy |
|---|---|
| `No module named ase/numpy/PIL/scipy/yaml` | Activate the cloned repo `.venv` and install `pip install -r requirements.txt`. |
| `YAeHMOP bind not found` | `--prepare-only` does not require it; actual solver requires separately installed `bind`, exported `YAEHMOP_BIND` and `YAEHMOP_PARAM_FILE`. |
| CSV says `header must be exactly` | Use `x,band_1,band_2[,band_3,...]` without extra columns and sort energies at each x. Saved EHT `orbital_*` fixtures are for the plotter, not input references. |
| No PBE bands appear | This repository intentionally does not redistribute third-party ab initio reference files. The carbon samples contain saved physical **EHT** eigenvalues. |
| `Invalid or out-of-range plot box` | `--bbox L,T,R,B` must be within the **oriented** normalized raster pixels. Use `inspect_png.py` to inspect suggested bounds. |
| `N reliable curve colors were not found` | Supply distinct `--colors` or use exact CSV/manual annotations. Do not invent missing bands. |
| `Not a verified P5 bitplane` | A normal gray PNG is not the 8-bit layer format. Use `--mode colors`, `mono`, `auto` or numeric CSV. |
| `Invalid band gap` | The gapped semiconductor interface does not accept gapless Dirac/metal cases; use the separately designed carbon protocol. |
| `Invalid raster / multipage` | Export **one** still raster frame in 8-bit RGB; animated GIF and multipage TIFF have ambiguous scientific meaning. |
| Diamond/graphite are treated as 2D | Use the code version containing `periodic_dim=3`, with 3 lattice vectors and nonzero overlap radius in c. See `scripts/recompute_carbon.py`. |
| GitHub Actions are slow or use too much compute | Actions deliberately run lightweight source+I/O tests only; full physical fitting must be performed locally by users. |

If an image trace is visibly wrong, stop and investigate `extraction_qa.json` and the preview rather than feeding it to the solver. Attach a minimal synthetic or openly licensed reproducer and full error text when opening an issue. Do not upload proprietary/third-party DFT data without permission.
