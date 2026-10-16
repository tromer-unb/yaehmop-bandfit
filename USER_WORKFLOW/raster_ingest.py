#!/usr/bin/env python3
"""P8: validated image-byte ingestion with one canonical internal raster PNG.

Accept any static raster format decoded by the installed Pillow plugins,
not arbitrary extensions. Image bytes, not suffix, determine file type.
The visible pixel matrix is normalized to an 8-bit RGB PNG. Exact
legacy P5 bitplanes (one 8-bit channel) are kept in 'L' PNG mode
after their entire sparse-layer signature is verified; bitplane
payload is not converted to RGB or thresholded.

This is *file-format* normalization only. It does not infer axes, bands,
crossings or hidden DFT states. The final optimizer input is always
P5 N-bit layered PNG after independent extraction + QA.
"""
from __future__ import annotations
from pathlib import Path
from PIL import Image,ImageOps,ImageCms,UnidentifiedImageError
import numpy as np
import hashlib,json,io,os,struct

SUPPORTED_MODES=("auto","layered","colors","mono")
MAX_PIXELS=40_000_000
MAX_FILE_BYTES=120_000_000
MIN_SIDE=24
def _format_guard(path:Path):
    if not path.is_file():raise FileNotFoundError(path)
    count=path.stat().st_size
    if not 0<count<=MAX_FILE_BYTES:
        raise ValueError(f"Raster input size limit exceeded or empty ({count} bytes)")
    return count

def normalize_raster(source,output,*,n_bands=None,expect_layered=False,
                     require_source_size=False):
    """Return (canonical_PNG_path, metadata_dict).

    Pixels larger than MAX_PIXELS and animated/multi-page inputs fail closed.
    The original source stays unmodified; EXIF orientation is applied and
    logged (pixel bbox coordinates must refer to the canonical oriented
    image). PNG output contains no user supplied EXIF, filename or arbitrary
    metadata. Embedded ICC is converted to sRGB when possible; invalid
    ICC inputs are rejected rather than discarded without disclosure.

    TIFF 16-bit/float/grayscale with greater than 8-bit precision is
    intentionally rejected: silent quantization can move line positions.
    """
    src=Path(source).expanduser().resolve();dst=Path(output).expanduser().resolve()
    if src==dst:raise ValueError("Canonical PNG output must not overwrite original raster")
    nbytes=_format_guard(src)
    if n_bands is not None and n_bands not in (2,4,6,8):
        raise ValueError("n_bands must be 2,4,6,8")
    try:
        # PNG requires verify() immediately after open, before getexif()
        # or any other method that can trigger pixel decoding.
        with Image.open(src) as verified:
            verified.verify()
        with Image.open(src) as f:
            format_name=f.format
            nframes=getattr(f,"n_frames",1)
            dims=f.size
            mode=f.mode
            orientation=f.getexif().get(274,1)
            icc=bool(f.info.get("icc_profile"))
            if nframes!=1:
                raise ValueError("Animated or multi-page raster rejected: use a single selected still frame")
            if format_name is None or format_name in ("PDF",):
                raise ValueError("Input is not a supported static raster image")
            w,h=dims
            if min(w,h)<MIN_SIDE or w*h>MAX_PIXELS:
                raise ValueError(f"Invalid raster dimensions {w}x{h}; require >=24px sides and <=40M pixels")
        with Image.open(src) as f:
            f.load()  # strict PIL decoding; do not enable truncated images
            if f.mode in ("I","I;16","I;16B","I;16L","F"):
                raise ValueError(f"High-precision raster mode {f.mode} cannot be silently converted to RGB8")
            if expect_layered and orientation not in (1,None):
                raise ValueError("Encoded P5 bitplanes cannot be EXIF-rotated")
            img=ImageOps.exif_transpose(f)
            # For modern true-embedded ICC, explicitly convert to canonical sRGB.
            if icc:
                try:
                    profile=ImageCms.ImageCmsProfile(io.BytesIO(f.info["icc_profile"]))
                    out_profile=ImageCms.createProfile("sRGB")
                    if img.mode not in ("RGB","RGBA","CMYK","L"):
                        img=img.convert("RGBA" if "transparency" in img.info else "RGB")
                    if img.mode=="RGBA":
                        rgb=ImageCms.profileToProfile(img.convert("RGB"),profile,out_profile,outputMode="RGB")
                        # Alpha doesn't define profile, preserve original alpha.
                        img=rgb.convert("RGBA");img.putalpha(ImageOps.exif_transpose(f).getchannel("A"))
                    else:
                        img=ImageCms.profileToProfile(img,profile,out_profile,outputMode="RGB")
                except Exception as exc:
                    raise ValueError("Invalid or unsupported embedded ICC profile; refusing implicit color shift") from exc
            legacy=False
            # Legacy multi-band encoding is *not* a normal grayscale image:
            # verify every bit-layer before calling it an original bitplane.
            if n_bands is not None and (not icc) and img.mode=="L":
                from generate_mc2d_multiband import FRAME,SIZE
                from image_trace_advanced import sparse_bitplane_signature
                legacy=sparse_bitplane_signature(np.asarray(img),n_bands,FRAME,SIZE)
            if expect_layered and not legacy:
                raise ValueError("This raster is not a verified P5 bitplane image; ordinary greyscale is not encoded bands")
            had_alpha=("A" in img.getbands() or "transparency" in img.info)
            if legacy:
                canonical=img.copy()  # gray-index values preserve 1<<band bitset
                normalized_mode="L_BITPLANES"
            else:
                if had_alpha:
                    rgba=img.convert("RGBA")
                    white=Image.new("RGBA",rgba.size,(255,255,255,255))
                    canonical=Image.alpha_composite(white,rgba).convert("RGB")
                else:
                    canonical=img.convert("RGB")
                normalized_mode="RGB8"
            if canonical.mode not in ("L","RGB") or canonical.size[0]*canonical.size[1]>MAX_PIXELS:
                raise ValueError("Internal canonical image contract violated")
            dst.parent.mkdir(parents=True,exist_ok=True)
            # Save as PNG, regardless of input format, without source metadata.
            canonical.save(dst,format="PNG")
    except (UnidentifiedImageError,OSError,SyntaxError) as exc:
        raise ValueError(f"Unreadable or unsupported raster data in {src.name}: {exc}") from exc
    with Image.open(dst) as saved:
        saved.load()
        assert saved.format=="PNG"
        if normalized_mode=="L_BITPLANES":
            assert np.array_equal(np.asarray(saved),np.asarray(canonical))
        else:
            assert saved.mode=="RGB" and np.array_equal(np.asarray(saved),np.asarray(canonical))
    return dst,{
      "original_path":str(src),"original_sha256":hashlib.sha256(src.read_bytes()).hexdigest(),
      "original_bytes":nbytes,"detected_format":format_name,"original_mode":mode,
      "original_size_px":list(dims),"n_frames":nframes,
      "exif_orientation":orientation,"icc_converted_to_srgb":icc,
      "alpha_composited_over_white":bool(normalized_mode=="RGB8" and had_alpha),
      "canonical_path":str(dst),"canonical_format":"PNG",
      "canonical_mode":normalized_mode,
      "canonical_size_px":list(canonical.size),
      "canonical_sha256":hashlib.sha256(dst.read_bytes()).hexdigest(),
      "legacy_P5_bitplanes_verified":legacy,
      "content_detection_not_file_extension":True,
      "scientific_warning":"Raster normalization is not a validation of DFT bands, k paths, gaps or degeneracies."
    }

def normalize_and_write_qa(source,outdir,*,n_bands=None,expect_layered=False):
    outdir=Path(outdir).resolve();outdir.mkdir(parents=True,exist_ok=True)
    image,meta=normalize_raster(source,outdir/"normalized_source.png",
             n_bands=n_bands,expect_layered=expect_layered)
    (outdir/"raster_ingest_qa.json").write_text(json.dumps(meta,indent=2)+"\n")
    return image,meta
