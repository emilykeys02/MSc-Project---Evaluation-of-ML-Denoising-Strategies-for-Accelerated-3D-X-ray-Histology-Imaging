"""
Normalise reconstructed volumes to [0, 1] before Dragonfly training, and
convert the denoised output back to the original intensity range afterwards.

Dragonfly's deep-learning tools expect input in a [0, 1] range, but a
reconstructed micro-CT volume is float32 in attenuation units (roughly -400 to
2700 here). Both steps are needed:

    normalise:    norm = (data - min) / (max - min)        clipped to [0, 1]
    denormalise:  data = norm * (max - min) + min

The same min and max must be used for both, so `normalise` writes them to a
small JSON file that `denormalise` can read back. The range is measured from a
middle band of slices only: the first and last slices are mostly air and
contain reconstruction artefacts that would stretch the range and compress the
tissue into a narrow part of it.

Volumes are processed one slice at a time, so a ~29 GB file is never held in
memory.

Usage:
    python normalise_volumes.py normalise   --input scan.raw --output scan_norm.raw
    python normalise_volumes.py denormalise --input denoised_norm.raw --output denoised.raw --ranges scan_norm_range.json
    python normalise_volumes.py denormalise --input denoised_norm.raw --output denoised.raw --min -165.62 --max 5119.59
"""

import argparse
import json
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# DEFAULTS -- change to match the volume being processed
# ---------------------------------------------------------------------------
DEFAULT_SHAPE = (2027, 1267, 2850)     # (Z, Y, X): slices, height, width
DEFAULT_DTYPE = np.float32
DEFAULT_SLICE_RANGE = (500, 2100)      # slices used to measure min/max (excludes the air at each end)


def slice_reader(path: Path, shape: tuple, dtype, start: int = 0, stop: int = None):
    """Yields (index, slice) one Z-slice at a time, without loading the volume."""
    n_slices, height, width = shape
    bytes_per_slice = height * width * np.dtype(dtype).itemsize
    stop = n_slices if stop is None else min(stop, n_slices)

    with open(path, "rb") as handle:
        handle.seek(bytes_per_slice * start)                  # jump straight to the first slice wanted
        for index in range(start, stop):
            data = np.frombuffer(handle.read(bytes_per_slice), dtype=dtype)
            yield index, data.reshape((height, width))


def measure_range(path: Path, shape: tuple, dtype, slice_range: tuple) -> tuple:
    """Finds the min and max intensity over the middle band of slices."""
    start, stop = slice_range
    vol_min, vol_max = np.inf, -np.inf
    for index, image in slice_reader(path, shape, dtype, start, stop):
        vol_min = min(vol_min, float(image.min()))
        vol_max = max(vol_max, float(image.max()))
        if (index + 1 - start) % 100 == 0:
            print(f"  scanned slice {index + 1}")
    return vol_min, vol_max


def normalise_volume(input_file, output_file, shape=DEFAULT_SHAPE, dtype=DEFAULT_DTYPE,
                     slice_range=DEFAULT_SLICE_RANGE) -> tuple:
    """Scales a volume to [0, 1] and returns the (min, max) used.

    The range comes from the middle slices only, but the scaling is applied to
    every slice, so values outside that band are clipped to 0 or 1. This is
    intended: it is what keeps bright specks from dominating the range.
    """
    input_path, output_path = Path(input_file), Path(output_file)
    print(f"\nNormalising {input_path.name}")
    print(f"  measuring range from slices {slice_range[0]}-{slice_range[1]}")
    vol_min, vol_max = measure_range(input_path, shape, dtype, slice_range)
    print(f"  original range: [{vol_min:.2f}, {vol_max:.2f}]")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as out:
        for index, image in slice_reader(input_path, shape, dtype):
            scaled = np.clip((image - vol_min) / (vol_max - vol_min), 0, 1)   # clip: values outside the band
            out.write(scaled.astype(dtype).tobytes())                          # append this slice to the file
            if (index + 1) % 200 == 0:
                print(f"  written slice {index + 1}/{shape[0]}")

    range_path = output_path.with_name(output_path.stem + "_range.json")
    range_path.write_text(json.dumps({"min": vol_min, "max": vol_max}, indent=2))  # so denormalise can reuse them
    print(f"  saved {output_path}\n  saved range to {range_path}")
    return vol_min, vol_max


def denormalise_volume(input_file, output_file, min_val: float, max_val: float,
                       shape=DEFAULT_SHAPE, dtype=DEFAULT_DTYPE) -> None:
    """Scales a [0, 1] volume back to its original intensity range.

    min_val and max_val must be the ones used when normalising, otherwise the
    output will not be comparable with the other reconstructions.
    """
    input_path, output_path = Path(input_file), Path(output_file)
    print(f"\nDenormalising {input_path.name} to [{min_val:.2f}, {max_val:.2f}]")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as out:
        for index, image in slice_reader(input_path, shape, dtype):
            restored = image * (max_val - min_val) + min_val    # exact inverse of the scaling above
            out.write(restored.astype(dtype).tobytes())
            if (index + 1) % 200 == 0:
                print(f"  written slice {index + 1}/{shape[0]}")
    print(f"  saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["normalise", "denormalise"])
    parser.add_argument("--input", required=True, help="Input .raw volume")
    parser.add_argument("--output", required=True, help="Output .raw volume")
    parser.add_argument("--shape", type=int, nargs=3, default=list(DEFAULT_SHAPE),
                        metavar=("Z", "Y", "X"), help=f"Volume shape (default {DEFAULT_SHAPE})")
    parser.add_argument("--slice-range", type=int, nargs=2, default=list(DEFAULT_SLICE_RANGE),
                        metavar=("START", "STOP"), help="Slices used to measure min/max (normalise only)")
    parser.add_argument("--ranges", default=None,
                        help="JSON file written by normalise, giving min and max (denormalise only)")
    parser.add_argument("--min", type=float, default=None, help="Original minimum (denormalise only)")
    parser.add_argument("--max", type=float, default=None, help="Original maximum (denormalise only)")
    args = parser.parse_args()

    shape = tuple(args.shape)
    if args.mode == "normalise":
        normalise_volume(args.input, args.output, shape=shape, slice_range=tuple(args.slice_range))
        return

    if args.ranges:                                            # prefer the saved file over typed values
        saved = json.loads(Path(args.ranges).read_text())
        min_val, max_val = saved["min"], saved["max"]
    elif args.min is not None and args.max is not None:
        min_val, max_val = args.min, args.max
    else:
        parser.error("denormalise needs either --ranges or both --min and --max")
    denormalise_volume(args.input, args.output, min_val, max_val, shape=shape)


if __name__ == "__main__":
    main()
