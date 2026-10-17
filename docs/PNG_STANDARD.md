# How ordinary PNGs become reproducible band references

The program supports two **different** concepts: the reader's ordinary image (a screenshot or scientific plot), and a calibrated *internal* multi-band image. They are not interchangeable.

## The input

A reference should specify:
- A real CIF with correct lattice constants and atom positions.
- A reciprocal k-path *from the actual source calculation*, in fractional coordinates of that CIF; k labels alone are not sufficient.
- Energy-axis ymin/ymax in eV at the actual bottom/top **plot frame**.
- Exactly the required identifiable valence and conduction curves: 1V+1C for --pairs 1, 2V+2C for --pairs 2, etc.
- If supplied as raster: actual frame rectangle L,T,R,B, curve colors and source image provenance.
- Treatment of SOC, spin, Fermi alignment, gap and DFT basis/method.

The program cannot infer electron filling, a Fermi surface or work-function alignment simply from an arbitrary screenshot.

## Recommended extraction paths

**Best: numerical CSV from the electronic-structure source**

A file must have exactly the columns x,band_1,band_2,...,band_N, ascending band energy per x, x strictly increasing from 0 to 1 and at least 15 finite rows. Example:
~~~csv
x,band_1,band_2
0.0,-1.15,1.22
0.1,-1.18,1.30
0.2,-1.27,1.21
~~~
This abbreviated snippet is explanatory only, not a complete valid numerical file.

~~~bash
python USER_WORKFLOW/prepare_band_png.py examples/synthetic/toy_2bands.csv \
  --n-pairs 1 --mode csv --emin -3.5 --emax 3.5 \
  --outdir outputs/csv_demo
~~~

**Good: clearly distinguishable RGB lines**

~~~bash
python USER_WORKFLOW/inspect_png.py YOUR_BANDS.png --outdir outputs/inspect

python USER_WORKFLOW/prepare_band_png.py YOUR_BANDS.png \
  --mode colors --n-pairs 1 --emin -6 --emax 6 \
  --bbox LEFT,TOP,RIGHT,BOTTOM \
  --colors '#RRGGBB,#RRGGBB' \
  --outdir outputs/prepared
~~~

Replace frame and hex codes by values measured in the **real** input; the placeholders are not valid numeric values. Color order must correspond to ascending band energies, and visible crossings/degeneracies must be verified.

**Limited: monochrome drawing**

The monochrome heuristic may reject too much occlusion. Manual annotations or an exact CSV are preferable to --allow-ambiguous. That flag allows an *approximation* but cannot resurrect invisible states.

## Internal 8-bit bitplane contract

The program generates bands_N_layers.png for N = 2, 4, 6 or 8. Each line is encoded in one bit plane:

- band 1 → bit value 1
- band 2 → bit value 2
- band 3 → bit value 4
- band 4 → bit value 8
- ... up to band 8 → bit value 128

At a crossing, two bits may be set in one pixel (e.g., band 1 + band 2 = value 3), preserving two *already extracted* curves. This representation does not recover overlapping lines that the original RGB screenshot did not reveal.

Conversion produces:
~~~text
reference_prepared/
  bands_2_layers.png       # actual internal representation
  extracted_curves.csv      # numerically calibrated energies
  extraction_preview.png    # visually overlay extraction onto source
  extraction_qa.json        # extraction mode, axis data, coverage, warnings, SHA
  raster_ingest_qa.json     # for ordinary raster input
~~~

Review preview, visible color assignment and gap before fitting. The encode_roundtrip_rmse_eV field tests only the new internal encoding, not the fidelity of the original digitization. All flags: [CLI reference](CLI_REFERENCE.md).

## Why a screenshot is not enough for some tasks

- A JPEG/PNG that shows only Γ–M–K–Γ cannot determine Fermi-surface topology over the entire BZ.
- Hidden or degenerate bands are not uniquely recoverable from a single-color raster.
- Ideal graphene, 1T-metal crossings, spin-polarized band structures and SOC splittings are **not** supported by the gapped PNG semiconductor objective.
- Physical comparison and transfer require the DFT reference to be generated in a compatible approximation, energy-zero convention, electron counting and reciprocal path.
- For a new material not included in the training parameter table, the code must reject unsupported orbitals rather than silently invent them.

The user never needs to construct a special PNG by hand: the code performs the conversion when the source is traceable, or raises an error and requests a reliable numerical reference.