# MoS₂ monolayer from a band picture (example of the graphical interface)

- `MoS2_PBE_2bands.png`: an ordinary band picture of the top valence (blue) and bottom conduction (red)
  band of a PBE calculation of the MoS₂ monolayer (FHI-aims, `validation/aims/MoS2_monolayer_pbe`), path
  Γ–K–K′–Γ (the K–K′ segment passes through M), energies relative to the midgap.
- `MoS2_monolayer.vasp`: the structure of that calculation.
- `MoS2_2bands.ini`: energy axis (−3 to 3.5 eV), k path, and the fit settings (1 VB + 1 CB, CMA-ES 4000 calls +
  Nelder–Mead polish, extra shells `auto`).

```bash
python USER_WORKFLOW/bands_fit.py examples/gui_mos2/MoS2_monolayer.vasp examples/gui_mos2/MoS2_2bands.ini \
    --outdir outputs/MoS2_picture
```

or, in `python band_fit_gui.py`, task *Fit to image*: *Load INI* with this file, choose the structure and the
picture, *Preview*, *Run fit*. Result (about 1 min on 24 threads): RMSE 0.074 eV, gap error 0.001 eV; the EHT
bands are drawn over the picture in `bands_on_original.png`.
