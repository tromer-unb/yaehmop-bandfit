# Band-image ingestion, digitization and validation

The CLI accepts a **static raster image that Pillow can safely decode**, independent of its filename extension. Supported reader formats normally include PNG, JPEG/JPG, 8-bit TIFF, BMP, WebP, single-frame GIF and PPM. The file is recognized by its actual decoded content, not just its suffix. CSV numerical data are handled directly instead; see [Numerical input](NUMERIC_BANDS.md).

## What happens to a PNG, JPEG or another raster?

```text
Source image on disk                     (never overwritten)
     │
     ▼
Decode + validate
 - actual file format, 1 frame only, image bounds and size
 - EXIF orientation, embedded color profile and transparency
 - SHA-256 checksum of original content
     │
     ▼
normalized_source.png                    (RGB8 for ordinary plots)
raster_ingest_qa.json                    (provenance, mode and hashes)
     │
     ▼
Find visible curves
 - colors: user-specified distinct hex RGB per curve
 - auto: conservative detection, rejects ambiguous colored plots
 - mono: single-color heuristic, requires manual scrutiny
 - layered: ONLY if verified existing sparse per-band bitplanes
     │
     ▼
Extracted x∈[0,1] and N energy curves   (N=2,4,6,8, sorted ascending)
Check per-band coverage, gaps, interpolation, round-trip accuracy
     │
     ▼
bands_N_layers.png                       (8-bit grayscale, special N-bit pixels)
extracted_curves.csv                      (resampled numeric curves)
extraction_qa.json                        (warnings and full statistics)
extraction_preview.png                    (manual review — mandatory)
     │
     ▼
Fit against actual YAeHMOP               (only on explicit fitting command)
```

An ordinary grayscale photograph is **not** assumed to contain original band-specific bitplanes. Invalid/ambiguous raster images, multipage/animated files, truncated data, high-precision >8-bit or float images requiring silent quantization, very large images (>40M pixels or >120MB), images with invalid embedded ICC profiles, and paths outside valid coordinates are rejected. A requested file-format conversion is never treated as proof of correct scientific band identification.

## The 8-bit bitplane PNG contract

The final `bands_N_layers.png` is a **single-channel, 8-bit PNG** with the selected curves in **different binary bits**:

| Pixel value | Meaning for N=4 |
|---:|---|
| 0 | No band at pixel |
| 1 (`2^0`) | Band 1 only (lowest energy) |
| 2 (`2^1`) | Band 2 only |
| 4 (`2^2`) | Band 3 only |
| 8 (`2^3`) | Band 4 only |
| 5 (`1+4`) | Bands 1 and 3 at the **same pixel** |

The encoding has a fixed **3200 × 3400** pixel canvas and an active plot box `(160,100,3000,3260)`, consistent with the original P5 scientific `read_colored_png` decoder. One byte per pixel supports at most **8 bands**. Each original selected band must cover a sufficient fraction of the k path; the decoder can preserve exact digital crossing locations without inventing shifts. It cannot infer physically hidden bands in raster pictures.

Use `--pairs 1` for N=2, `--pairs 2` for N=4, `--pairs 3` for N=6 and `--pairs 4` for N=8. These are **separately selected and separately optimized windows**, not always an evaluation of 8 underlying bands.

## Quick commands

```bash
# Suggest image bounds and dominant trace colors (no YAeHMOP):
python USER_WORKFLOW/inspect_png.py examples/synthetic/toy_2bands.png \
  --outdir outputs/inspect

# Convert 2 visible distinct colors into 2 independent layer bits:
python USER_WORKFLOW/prepare_band_png.py examples/synthetic/toy_2bands.png \
  --n-pairs 1 --emin -3.5 --emax 3.5 \
  --mode colors --bbox 115,105,1210,820 \
  --colors '#c82d37,#1c7ad5' --color-tolerance 36 \
  --outdir outputs/colored

# Numeric CSV avoids raster ambiguity:
python USER_WORKFLOW/prepare_band_png.py examples/synthetic/toy_4bands.csv \
  --n-pairs 2 --emin -3.5 --emax 3.5 \
  --mode csv --outdir outputs/numeric
```

To supply a JPEG instead, replace the input filename and verify that color compression did not break tracing. When curves blend or have near-identical colors, prefer a numerical CSV (not a forced trace). `--mode auto` may infer colors, but do not consider its decision authoritative without inspecting the QA image. For a raster missing a clear coordinate frame, supply an exact `--bbox L,T,R,B`.

## Quality checks and careful interpretation

Inspect all of these before a physics run:
- `raster_ingest_qa.json`: detected format, original and normalized hashes, dimensions, transparency, orientation, color-profile transformation, bitplane validation status.
- `extraction_preview.png`: source RGB plot and selected/reconstructed bands overlaid; check valence/conduction ordering, node positions and interpolation visually.
- `extraction_qa.json`: mode chosen, number of selected bands, gap, coverage, unsafe extrapolation or ambiguity flags, `encode_roundtrip_rmse_eV`.
- `extracted_curves.csv`: energy samples that will actually be sent to the fitter.

A low **encode/decode round-trip** error measures the encoding pipeline only. It does **not** validate DFT accuracy, branch identity at crossings, real Dirac nodes, basis completeness, reference gap, or out-of-sample EHT transfer.

## Manual digitization

If `--mode colors` is unreliable, use `USER_WORKFLOW/digitize_bands.py` to create a template for manually specified pixel anchors (`--template`), optionally use the interactive editor (`--interactive`) on a machine with a display, and convert using `--annotations ... --convert`. Call `--help` for each exact syntax. Manual annotations bind the **original** and **normalized** raster SHA-256 hashes; they must not be silently reused for a different input.

**EXIF orientation:** the manual anchor coordinates apply to the **normalized, oriented** PNG, not to the raw JPEG coordinates before rotation.

## Limitations

A source plot may omit upper/lower curves or hide overlapping states. Automatic extraction of four colors from a two-color plot cannot produce four genuine reference bands. The safer choices are the author's original numerical DFT/PBE band table, manually verified curve anchors, or a smaller truly available N-band fit. The gapped fitting objective rejects an apparent frontier gap below about 0.02 eV; metallic and graphene gapless cases require a distinct objective, even though numerical EHT spectra can be plotted separately.
