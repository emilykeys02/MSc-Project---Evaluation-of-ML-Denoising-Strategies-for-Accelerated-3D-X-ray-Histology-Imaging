"""
Convert denoised projections back from linear attenuation space to raw
transmission values, so they can be reconstructed by Nikon CT Pro.

This is the exact inverse of linearise_projections.py, and is needed because
CT Pro expects raw projections and applies its own flat-field correction, log
transform and beam-hardening correction internally as part of reconstruction.
Feeding it already-linearised data would apply those steps twice.

    p      = -log(I_norm)            ->  I_norm = exp(-p)
    I_norm = raw / white_level       ->  raw    = I_norm * white_level
    (or, with a flat and dark image:     raw    = I_norm * (flat - dark) + dark)

Use the same flat/dark or white level here as was used to linearise, otherwise
the intensity scale will not match the original scan.

Not needed if reconstruction is done with a pipeline that accepts attenuation
data directly (ASTRA, TIGRE, or a custom FDK implementation).

Usage examples:
    python delinearise_projections.py --lin_dir ./denoised_linear --white_level 60000 --out_dir ./denoised_raw
    python delinearise_projections.py --lin_dir ./denoised_linear --flat ./flat.tif --dark ./dark.tif --out_dir ./denoised_raw

    # rename the output, e.g. Inference_unet_67ms_0001.tif, _0002.tif, ...
    python delinearise_projections.py --lin_dir ./denoised_linear --white_level 60000 \
        --out_dir ./denoised_raw --save_name Inference_unet_67ms_
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile as tiff


def load_image(path) -> np.ndarray:
    """Reads a TIFF as float32; the exponential below needs floating point."""
    return tiff.imread(str(path)).astype(np.float32)


def delinearise(p: np.ndarray, flat: np.ndarray, dark: np.ndarray) -> np.ndarray:
    """Undoes the log and the flat-field correction for one projection."""
    i_norm = np.exp(-p)                                      # back to normalised transmission (0-1)
    if dark is not None:
        return i_norm * (flat - dark) + dark                 # restore the dark offset that was subtracted
    return i_norm * flat                                     # scalar white level, or flat-only correction


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lin_dir", required=True, help="Folder of linearised (denoised) TIFFs")
    parser.add_argument("--out_dir", required=True, help="Folder to write raw-style TIFFs")
    parser.add_argument("--flat", default=None, help="Open-beam TIFF used when linearising")
    parser.add_argument("--dark", default=None, help="Dark-field TIFF used when linearising")
    parser.add_argument("--white_level", type=float, default=None,
                        help="Scalar open-beam value used when linearising")
    parser.add_argument("--dtype", default="uint16", choices=["uint16", "float32"],
                        help="Output type (default uint16, matching the detector's raw output)")
    parser.add_argument("--pattern", default="*.tif*", help="Glob pattern for input files")
    parser.add_argument("--save_name", default=None,
                        help="Output filename prefix; numbering is appended. Keeps input names if omitted.")
    parser.add_argument("--start_index", type=int, default=1, help="First number for --save_name")
    parser.add_argument("--num_digits", type=int, default=4, help="Zero-padding for --save_name")
    args = parser.parse_args()

    if args.white_level is None and args.flat is None:
        sys.exit("Provide either --white_level or --flat, matching what was used to linearise")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lin_paths = sorted(Path(args.lin_dir).glob(args.pattern))   # sorted: keeps projection order
    if not lin_paths:
        sys.exit(f"No files matching {args.pattern} found in {args.lin_dir}")
    print(f"Found {len(lin_paths)} linearised projections")

    dark = load_image(args.dark) if args.dark else None
    flat_img = load_image(args.flat) if args.flat else None

    for i, path in enumerate(lin_paths):
        p = load_image(path)
        flat = (np.full_like(p, args.white_level, dtype=np.float32)
                if args.white_level is not None else flat_img)
        raw = delinearise(p, flat, dark)

        if args.dtype == "uint16":
            raw = np.clip(raw, 0, 65535).astype(np.uint16)   # clip before casting, or values wrap around
        else:
            raw = raw.astype(np.float32)

        out_name = (f"{args.save_name}{i + args.start_index:0{args.num_digits}d}.tif"
                    if args.save_name else path.name)
        tiff.imwrite(str(out_dir / out_name), raw)
        if i % 50 == 0 or i == len(lin_paths) - 1:
            print(f"[{i + 1}/{len(lin_paths)}] {out_name} -> range [{raw.min()}, {raw.max()}]")

    print(f"\nDone. Raw-style projections written to {out_dir}, ready for reconstruction.")


if __name__ == "__main__":
    main()
