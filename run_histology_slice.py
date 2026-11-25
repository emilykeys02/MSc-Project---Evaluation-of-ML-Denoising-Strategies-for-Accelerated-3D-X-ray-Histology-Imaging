"""
Extract a single histology-orientation slice from a reconstructed volume.

Runs the reslice pipeline (histreslice_pipeline.py) with the rotation angles
and crop boxes already measured for this sample, then saves one slice as a
TIFF. Running it on every exposure and every denoised output gives slices in
exactly the same plane, so they can be compared side by side.

The defaults below are the values solved for the lung sample used in this
study; for a different sample, measure new ones and pass them as flags.

Usage:
    # same sample, a different denoised volume
    python run_histology_slice.py input.vol output.tif --slice 390

    # find the settings for a new sample: save the whole stack, then measure
    # the rectangle and slice number in Fiji
    python run_histology_slice.py input.vol preview.tif --preview

    # a new sample, with its own geometry
    python run_histology_slice.py input.vol output.tif \
        --width 1800 --height 1400 --slices 3000 \
        --rotz -90.5 --roty -160.2 \
        --window 700 2200 300 1750 500 1200 \
        --wax-rect 60 200 1500 420 --slice 300
"""

import argparse

import numpy as np
import tifffile as tiff

from histreslice_pipeline import process_volume

# ---------------------------------------------------------------------------
# DEFAULTS for this sample (angles measured once in Fiji, then reused)
# ---------------------------------------------------------------------------
DEFAULT_ROTATE_Z = -92.0256                                  # degrees, in-plane rotation
DEFAULT_ROTATE_Y = -166.7048                                 # degrees, after the first reslice
DEFAULT_WINDOW_VOLUME = (633, 2133, 294, 1693, 486, 1146)    # z_lo z_hi x_lo x_hi y_lo y_hi
DEFAULT_WAX_RECT = (57, 214, 1557, 437)                      # x y w h, before the margin is added
DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_SLICES = 2027, 1267, 2850
DEFAULT_SLICE_INDEX = 390                                    # the histology-matched slice


def save_tiff(path, image):
    """Saves a float32 TIFF with a sensible display range for Fiji.

    The 1st and 99th percentiles are stored as the display min/max so the image
    opens with usable contrast instead of being stretched by a few extreme
    voxels. The pixel values themselves are unchanged.
    """
    low, high = np.percentile(image, [1, 99])
    tiff.imwrite(path, image.astype(np.float32), imagej=True,
                 metadata={"min": float(low), "max": float(high)})


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_path", help="Raw .vol/.raw input volume")
    parser.add_argument("output_path", help="Where to save the slice (.tif)")
    parser.add_argument("--slice", type=int, default=DEFAULT_SLICE_INDEX,
                        help=f"Slice to extract (default {DEFAULT_SLICE_INDEX})")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--slices", type=int, default=DEFAULT_SLICES)
    parser.add_argument("--rotz", type=float, default=DEFAULT_ROTATE_Z,
                        help=f"Rotation about Z in degrees (default {DEFAULT_ROTATE_Z})")
    parser.add_argument("--roty", type=float, default=DEFAULT_ROTATE_Y,
                        help=f"Rotation about Y in degrees (default {DEFAULT_ROTATE_Y})")
    parser.add_argument("--window", type=int, nargs=6, default=list(DEFAULT_WINDOW_VOLUME),
                        metavar=("Z_LO", "Z_HI", "X_LO", "X_HI", "Y_LO", "Y_HI"),
                        help="Rough box around the sample, in raw volume coordinates")
    parser.add_argument("--wax-rect", type=int, nargs=4, default=list(DEFAULT_WAX_RECT),
                        metavar=("X", "Y", "W", "H"),
                        help="Wax-block rectangle, measured after the rotations")
    parser.add_argument("--margin", type=int, default=20,
                        help="Margin in pixels added around the wax rectangle (default 20)")
    parser.add_argument("--preview", action="store_true",
                        help="Save the whole stack before the wax crop, to measure a new sample")
    args = parser.parse_args()

    x, y, w, h = args.wax_rect
    wax_rect = (x - args.margin, y - args.margin,           # grow the rectangle by the margin on all sides
                w + 2 * args.margin, h + 2 * args.margin)
    if args.preview:
        wax_rect = (0, 0, 100_000, 100_000)                 # oversized: keeps everything, clipped to the volume

    print(f"Processing {args.input_path}")
    print(f"  dims {args.width}x{args.height}x{args.slices}, rotz={args.rotz}, roty={args.roty}")
    print(f"  window={tuple(args.window)}")
    print("  PREVIEW: saving the full stack before the wax crop" if args.preview
          else f"  wax rectangle with margin = {wax_rect}")

    stack, _ = process_volume(
        args.input_path, args.width, args.height, args.slices,
        args.rotz, args.roty,
        wax_crop_rect=wax_rect,
        window_volume=tuple(args.window),
        is_32bit=True, pad=25, rotate_order=1, rotate90_clockwise=True,
        final_reslice=False,        # leave Y as the paging axis, so --slice indexes histology planes
    )
    print("Resliced stack shape (Y, Z, X):", stack.shape)

    if args.preview:
        save_tiff(args.output_path, stack)
        print(f"Saved preview stack -> {args.output_path}")
        print("Open it in Fiji, find the slice you want, draw a rectangle around the tissue, "
              "and read x/y/width/height from Edit > Selection > Specify for --wax-rect.")
        return

    if args.slice >= stack.shape[0]:
        raise ValueError(f"Slice {args.slice} is out of range: the stack has {stack.shape[0]} slices.")

    save_tiff(args.output_path, stack[args.slice])
    print(f"Saved slice {args.slice} -> {args.output_path}")


if __name__ == "__main__":
    main()
