"""
Convert raw micro-CT projections into linear attenuation space before denoising.

Raw projections are transmission images: bright where the beam passes freely,
dark where it is absorbed, and the relationship between material thickness and
pixel value is exponential. Denoising models and reconstruction both assume a
linear relationship, so each projection is flat-field corrected and then
log-transformed:

    I_norm = (raw - dark) / (flat - dark)     # correct for beam and detector non-uniformity
    p      = -log(I_norm)                     # Beer-Lambert: p is now proportional to
                                              # the attenuation along each ray

The output has a background near 0, with denser material giving higher values.

The open-beam (flat) reference can be supplied three ways:
  --flat       a measured open-beam image (best, if one was acquired)
  --white_level  a single number (Nikon/Xtek scanners apply their own shading
                 correction onboard, so the scalar WhiteLevel from the .xtekct
                 file is usually enough)
  --estimate_flat_roi  estimate it from a background-only strip of the detector
                 when neither of the above exists, as in this study

Usage examples:
    python linearise_projections.py --proj_dir ./projections --flat ./flat.tif --out_dir ./projections_linear
    python linearise_projections.py --proj_dir ./projections --white_level 60000 --out_dir ./projections_linear
    python linearise_projections.py --proj_dir ./projections --estimate_flat_roi 0 0 2027 50 --out_dir ./projections_linear
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile as tiff


def load_image(path) -> np.ndarray:
    """Reads a TIFF as float32 so the division and log below are not done in integers."""
    return tiff.imread(str(path)).astype(np.float32)


def estimate_flat_from_roi(proj_paths: list, roi: tuple) -> np.ndarray:
    """Estimates a flat field from a background-only region, when no flat image exists.

    roi = (x0, y0, x1, y1), a region that contains no sample in any projection
    (e.g. a strip along the top of the detector). The mean level there is taken
    as the open-beam value and broadcast to a full frame. This assumes the
    background is reasonably uniform across the detector; a strong spatial
    gradient would need a measured flat field instead.
    """
    x0, y0, x1, y1 = roi
    step = max(1, len(proj_paths) // 20)                   # sample ~20 projections, not all of them
    values = [load_image(p)[y0:y1, x0:x1].mean() for p in proj_paths[::step]]
    background_level = float(np.mean(values))
    print(f"Estimated flat background level: {background_level:.3f}")
    return np.full_like(load_image(proj_paths[0]), background_level, dtype=np.float32)


def linearise(raw: np.ndarray, flat: np.ndarray, dark: np.ndarray, eps: float) -> np.ndarray:
    """Applies flat-field correction and the negative log to one projection."""
    numerator = raw - dark if dark is not None else raw
    denominator = flat - dark if dark is not None else flat

    denominator = np.clip(denominator, eps, None)          # guard against divide-by-zero
    i_norm = np.clip(numerator / denominator, eps, None)   # and against log(0) or log(negative)
    return (-np.log(i_norm)).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--proj_dir", required=True, help="Folder of raw projection TIFFs")
    parser.add_argument("--out_dir", required=True, help="Folder to write linearised TIFFs")
    parser.add_argument("--flat", default=None, help="Open-beam reference TIFF")
    parser.add_argument("--dark", default=None, help="Dark-field (no-beam) reference TIFF")
    parser.add_argument("--estimate_flat_roi", nargs=4, type=int, default=None,
                        metavar=("X0", "Y0", "X1", "Y1"),
                        help="Background-only ROI to estimate the flat field from")
    parser.add_argument("--white_level", type=float, default=None,
                        help="Scalar open-beam value, e.g. WhiteLevel from the .xtekct file")
    parser.add_argument("--eps", type=float, default=1e-4,
                        help="Lower clip before dividing and taking the log (default 1e-4)")
    parser.add_argument("--pattern", default="*.tif*", help="Glob pattern for input files")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    proj_paths = sorted(Path(args.proj_dir).glob(args.pattern))   # sorted: keeps projection order
    if not proj_paths:
        sys.exit(f"No files matching {args.pattern} found in {args.proj_dir}")
    print(f"Found {len(proj_paths)} projections")

    dark = load_image(args.dark) if args.dark else None
    flat = None
    if args.white_level is not None:
        print(f"Using scalar white level: {args.white_level}")   # flat built per projection below
    elif args.flat:
        flat = load_image(args.flat)
    elif args.estimate_flat_roi:
        flat = estimate_flat_from_roi(proj_paths, args.estimate_flat_roi)
    else:
        sys.exit("Provide one of --white_level, --flat, or --estimate_flat_roi")

    for i, path in enumerate(proj_paths):
        raw = load_image(path)
        frame_flat = (np.full_like(raw, args.white_level, dtype=np.float32)
                      if args.white_level is not None else flat)
        linear = linearise(raw, frame_flat, dark, args.eps)
        tiff.imwrite(str(out_dir / path.name), linear)      # same filename, so pairing stays simple
        if i % 50 == 0 or i == len(proj_paths) - 1:
            print(f"[{i + 1}/{len(proj_paths)}] {path.name} -> range [{linear.min():.3f}, {linear.max():.3f}]")

    print(f"\nDone. Linearised projections written to {out_dir}")


if __name__ == "__main__":
    main()
