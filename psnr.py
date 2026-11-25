"""
Whole-image PSNR of every reconstructed volume against the clean reference.

For each volume in the registry, a set of evenly spaced Z-slices is compared
with the same slices of the 354ms-Optimal reference scan:

    MSE  = mean( (reference - test)^2 )   over every pixel of the slice
    PSNR = 20 * log10( MAX_I / sqrt(MSE) )

MAX_I is measured at run time from the intensity range OUTSIDE the specimen
(cassette, mount, background). The lung sample contains calcifications that
are far brighter than tissue, and including them would inflate MAX_I, and so
PSNR, for every comparison. Set MAX_I_MODE = "fixed" to use the hard-coded
REFERENCE_DATA_RANGE instead.

Volumes are read with np.memmap, so only the requested slices are pulled off
disk and the ~29 GB per volume is never loaded into memory. Run this on the
machine where the M: network drive is mounted.

Outputs two CSV files: one row per slice, and one row per volume (mean PSNR).

Usage:
    python psnr_3d.py                # full run
    python psnr_3d.py --preview      # save one slice per volume as PNG first,
                                     # to check the axis order and byte order
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from volume_registry import (
    NX, NY, NZ, VOLUMES, REFERENCE_KEY, get_dtype, resolve_path,
)

# ---------------------------------------------------------------------------
# 1. SETTINGS
# ---------------------------------------------------------------------------
Z_START, Z_END, Z_STEP = 650, 1415, 10   # sampled slices: every 10th over the whole specimen (~77 slices)
                                         # Z_STEP = 1 uses all 766 slices (most rigorous, slowest)

# Specimen bounding box in X/Y, excluded when measuring MAX_I so that
# calcifications inside the sample cannot inflate it. Z is not restricted
# because the specimen spans the whole sampled Z range above.
SPECIMEN_X_MIN, SPECIMEN_X_MAX = 600, 1350
SPECIMEN_Y_MIN, SPECIMEN_Y_MAX = 790, 1400

MAX_I_MODE = "auto_background"           # "auto_background" (recommended) or "fixed"
REFERENCE_DATA_RANGE = 2683.96 - (-347.171)   # full min/max span of the reference, used when MAX_I_MODE = "fixed"

OUTPUT_DETAIL_CSV = Path("psnr_per_slice.csv")
OUTPUT_SUMMARY_CSV = Path("psnr_summary.csv")
PREVIEW_DIR = Path("psnr_preview")


# ---------------------------------------------------------------------------
# 2. READING SLICES
# ---------------------------------------------------------------------------
def open_volume(path: Path, dtype: np.dtype) -> np.memmap:
    """Maps the file without reading it; slices are fetched on demand."""
    return np.memmap(path, dtype=dtype, mode="r", shape=(NZ, NY, NX))


def read_slice(volume: np.memmap, z: int) -> np.ndarray:
    """Reads one Z-slice as float64, so the squared differences stay accurate."""
    return np.asarray(volume[z], dtype=np.float64)


def background_mask() -> np.ndarray:
    """True for pixels outside the specimen box, i.e. the region MAX_I is measured from."""
    mask = np.ones((NY, NX), dtype=bool)
    mask[SPECIMEN_Y_MIN:SPECIMEN_Y_MAX, SPECIMEN_X_MIN:SPECIMEN_X_MAX] = False
    return mask


# ---------------------------------------------------------------------------
# 3. PSNR
# ---------------------------------------------------------------------------
def psnr_from_mse(mse: float, max_i: float) -> float:
    """PSNR in dB; infinite when the two slices are identical."""
    return float("inf") if mse <= 0 else 20.0 * np.log10(max_i / np.sqrt(mse))


def compare_volumes(ref_slices: dict, test_volume, z_values, max_i: float) -> list:
    """Returns [(z, mse, psnr), ...] for one test volume.

    ref_slices is pre-read once by main() so the reference isn't re-read from
    the network drive for every test volume.
    """
    rows = []
    for z in z_values:
        test_slice = read_slice(test_volume, z)
        mse = float(np.mean((ref_slices[z] - test_slice) ** 2))   # pixel-wise MSE for this slice
        rows.append((z, mse, psnr_from_mse(mse, max_i)))
    return rows


def read_reference(ref_volume, z_values) -> tuple:
    """Reads every reference slice once, and measures the background intensity range.

    Returns (slices, background_range, whole_slice_range).
    """
    mask = background_mask()
    slices = {}
    whole_min, whole_max = np.inf, -np.inf
    bg_min, bg_max = np.inf, -np.inf
    for z in z_values:
        s = read_slice(ref_volume, z)
        slices[z] = s
        whole_min, whole_max = min(whole_min, s.min()), max(whole_max, s.max())
        background = s[mask]                                   # specimen excluded
        bg_min, bg_max = min(bg_min, background.min()), max(bg_max, background.max())
    return slices, float(bg_max - bg_min), float(whole_max - whole_min)


# ---------------------------------------------------------------------------
# 4. PREVIEW -- check axis order and byte order before a full run
# ---------------------------------------------------------------------------
def run_preview():
    """Saves the reference slice and one denoised slice as PNGs.

    If either looks garbled or striped, the (NZ, NY, NX) order or the
    byte order for that file is wrong.
    """
    import matplotlib
    matplotlib.use("Agg")                                      # no display needed
    import matplotlib.pyplot as plt

    PREVIEW_DIR.mkdir(exist_ok=True)
    test_key = next(k for k, v in VOLUMES.items() if v.get("group") == "denoised")

    for key in (REFERENCE_KEY, test_key):
        info = VOLUMES[key]
        volume = open_volume(resolve_path(key, info), get_dtype(info))
        image = read_slice(volume, Z_START)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image, cmap="gray")
        ax.set_title(f"{key}  z={Z_START}  shape={image.shape}")
        ax.axis("off")
        out_path = PREVIEW_DIR / f"preview_{key}.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out_path}  (min={image.min():.2f}, max={image.max():.2f}, mean={image.mean():.2f})")


# ---------------------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------------------
def main():
    z_values = list(range(Z_START, Z_END + 1, Z_STEP))
    print(f"Sampling {len(z_values)} slices (z={Z_START}-{Z_END}, step={Z_STEP}), MAX_I_MODE={MAX_I_MODE!r}\n")

    ref_info = VOLUMES[REFERENCE_KEY]
    ref_volume = open_volume(resolve_path(REFERENCE_KEY, ref_info), get_dtype(ref_info))
    print(f"Reference: {REFERENCE_KEY}\nPre-reading reference slices...")
    ref_slices, bg_range, whole_range = read_reference(ref_volume, z_values)

    print(f"  Whole-slice range (includes calcifications): {whole_range:.4f}")
    print(f"  Background-only range (specimen box excluded): {bg_range:.4f}")
    max_i = bg_range if MAX_I_MODE == "auto_background" else REFERENCE_DATA_RANGE
    print(f"  -> MAX_I = {max_i:.4f}\n")

    detail_rows, summary_rows = [], []
    for key, info in VOLUMES.items():
        if key == REFERENCE_KEY:
            continue                                            # nothing to compare against itself
        start = time.time()
        try:
            path = resolve_path(key, info)
        except FileNotFoundError as error:
            print(f"  SKIP {key}: {error}")                      # missing file: report and carry on
            continue

        test_volume = open_volume(path, get_dtype(info))
        rows = compare_volumes(ref_slices, test_volume, z_values, max_i)
        detail_rows += [[key, info["group"], info["method"], z, mse, psnr] for z, mse, psnr in rows]

        finite = [psnr for _, _, psnr in rows if np.isfinite(psnr)]
        mean_psnr = float(np.mean(finite)) if finite else float("nan")
        summary_rows.append([key, info["group"], info["method"], len(z_values), mean_psnr])
        print(f"{key:32s} mean PSNR = {mean_psnr:7.3f} dB   ({time.time() - start:.1f}s)")

    for path, header, rows in [
        (OUTPUT_DETAIL_CSV, ["volume_key", "group", "method", "z_slice", "mse", "psnr_db"], detail_rows),
        (OUTPUT_SUMMARY_CSV, ["volume_key", "group", "method", "n_slices", "mean_psnr_db"], summary_rows),
    ]:
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)

    print(f"\nDone. Wrote {OUTPUT_DETAIL_CSV} and {OUTPUT_SUMMARY_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true",
                        help="Save one slice per volume as a PNG and exit.")
    if parser.parse_args().preview:
        run_preview()
    else:
        main()
