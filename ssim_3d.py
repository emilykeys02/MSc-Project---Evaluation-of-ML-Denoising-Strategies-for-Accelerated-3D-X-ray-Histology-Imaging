"""
3D structural similarity (SSIM) of denoised volumes against the clean reference,
measured over the cropped lung-sample region only.

For each comparison the script:
  1. loads the reference volume (354ms-Optimal) and one comparison volume,
     reading only the Z-slices inside the specimen bounding box;
  2. crops both to the same box, so background wax and air cannot dominate
     the score;
  3. computes 3D SSIM over that crop and appends the result to an Excel file.

Windows of 7x7x7 and 11x11x11 voxels are both reported, because the
microstructures of interest are only about 5-10 voxels across and the window
size affects how sensitive SSIM is to them.

One shared data range is used for every comparison, taken from the reference
scan. Each scan's own min/max would be wider at lower exposures (a few extreme
noisy voxels push it outward), which would bias SSIM in step with exposure --
exactly the confound this avoids.

Usage:
    python ssim_3d.py 354ms-Optimal 67ms          # one comparison
    python ssim_3d.py 354ms-Optimal denoised      # batch: every denoised output
    python ssim_3d.py 354ms-Optimal n2v-proj      # batch: one method, all exposures
    python ssim_3d.py                             # prompts for both volumes
A full file path can be given instead of a registered name.
"""

import sys
import time
from pathlib import Path

import numpy as np
from skimage.metrics import structural_similarity as ssim

from volume_registry import NX, NY, NZ, VOLUMES, resolve_path, select_volumes

# ---------------------------------------------------------------------------
# 1. SETTINGS
# ---------------------------------------------------------------------------
DTYPE = np.float32                     # voxel type of every volume

# Specimen bounding box, measured in ImageJ. SSIM is computed inside this box
# only, so the result reflects the tissue rather than the surrounding wax.
X_MIN, X_MAX = 768, 1265               
Y_MIN, Y_MAX = 1120, 1318
Z_MIN, Z_MAX = 847, 1091

KERNEL_SIZES = [7]
REFERENCE_DATA_RANGE = 131.94813537597656 - (-27.063596725463867)   # shared range, from the reference crop
EXCEL_OUTPUT_PATH = Path("./ssim_results.xlsx") #produces an output of an excel speadsheet of results


# ---------------------------------------------------------------------------
# 2. LOADING
# ---------------------------------------------------------------------------
def load_cropped_volume(path: Path, nz: int, byteswap: bool = False) -> np.ndarray:
    """Reads the Z-slices inside the bounding box and crops each to the X/Y box.

    Each slice is read as one contiguous block and cropped in memory rather
    than read as a strided region. The files sit on a network share, where a
    strided read means many small round-trips; one sequential read per slice
    transfers more bytes but is far faster in practice.

    byteswap: True for files written in the opposite byte order (the Med+Gaus
    outputs). Byte order changes only how the bytes are interpreted, not how
    many there are, so the header check below is unaffected. Reading with the
    wrong order does not raise an error, it silently produces NaNs.
    """
    itemsize = np.dtype(DTYPE).itemsize
    expected_bytes = NX * NY * nz * itemsize
    actual_bytes = path.stat().st_size

    if actual_bytes == expected_bytes:
        header_offset = 0                                   # headerless raw file
    elif actual_bytes > expected_bytes:
        header_offset = actual_bytes - expected_bytes       # skip the header that precedes the voxels
        print(f"[{path.name}] skipping {header_offset} byte header.")
    else:
        raise ValueError(
            f"[{path.name}] file is smaller than the expected {expected_bytes:,} bytes "
            f"({actual_bytes:,} found). Check NX/NY/NZ and the data type."
        )

    read_dtype = np.dtype(DTYPE).newbyteorder() if byteswap else np.dtype(DTYPE)
    slice_bytes = NX * NY * itemsize
    out = np.empty((Z_MAX - Z_MIN + 1, Y_MAX - Y_MIN + 1, X_MAX - X_MIN + 1), dtype=DTYPE)

    start = time.time()
    with open(path, "rb") as handle:
        for i, z in enumerate(range(Z_MIN, Z_MAX + 1)):
            handle.seek(header_offset + z * slice_bytes)    # jump to the start of this slice
            plane = np.frombuffer(handle.read(slice_bytes), dtype=read_dtype).reshape(NY, NX)
            out[i] = plane[Y_MIN:Y_MAX + 1, X_MIN:X_MAX + 1].astype(DTYPE)   # crop, and drop byte order
    print(f"  Loaded {out.shape[0]} slices in {time.time() - start:.1f}s"
          f"{', byte-swapped' if byteswap else ''}.")
    return out


def resolve_token(token: str) -> tuple:
    """Turns a registered name or a pasted file path into (display name, path, nz, byteswap).
    nz is the number of Z-slices in that file, needed for the header check
    above. It defaults to the standard NZ but can be set per entry, e.g. for
    an output that contains fewer slices than the full reconstruction.
    """
    if token in VOLUMES:
        info = VOLUMES[token]
        return token, resolve_path(token, info), info.get("nz", NZ), info.get("byteswap", False)
    path = Path(token)                                      # not a registered name: treat as a path
    return f"custom:{path.name}", path, NZ, False


# ---------------------------------------------------------------------------
# 3. SSIM
# ---------------------------------------------------------------------------
def compute_ssim(reference: np.ndarray, comparison: np.ndarray) -> dict:
    """3D SSIM at each window size in KERNEL_SIZES, over the whole crop."""
    scores = {}
    for size in KERNEL_SIZES:
        if size > min(reference.shape):
            print(f"  Window {size} exceeds the smallest crop dimension -- skipped.")
            continue
        scores[size] = ssim(reference, comparison,
                            data_range=REFERENCE_DATA_RANGE,   # shared range for every comparison
                            win_size=size, channel_axis=None)  # channel_axis=None: treat input as one 3D volume
    return scores


def run_single_comparison(ref_token: str, cmp_token: str) -> dict:
    """Loads both volumes, computes SSIM, and returns one result row."""
    ref_name, ref_path, ref_nz, ref_swap = resolve_token(ref_token)
    cmp_name, cmp_path, cmp_nz, cmp_swap = resolve_token(cmp_token)
    print(f"\nReference:  {ref_name}  ({ref_path})")
    print(f"Comparison: {cmp_name}  ({cmp_path})")

    reference = load_cropped_volume(ref_path, ref_nz, byteswap=ref_swap)
    comparison = load_cropped_volume(cmp_path, cmp_nz, byteswap=cmp_swap)
    assert reference.shape == comparison.shape, "Crops must have the same shape."

    scores = compute_ssim(reference, comparison)
    for size, score in scores.items():
        print(f"    SSIM ({size}x{size}x{size}): {score:.4f}")

    row = {"comparison": cmp_token}
    row.update({f"ssim_{size}": score for size, score in scores.items()})
    return row


# ---------------------------------------------------------------------------
# 4. OUTPUT
# ---------------------------------------------------------------------------
def save_results_to_excel(rows: list, output_path: Path = EXCEL_OUTPUT_PATH):
    """Writes results to a flat Excel table, updating rows that already exist.

    Re-running a comparison overwrites its previous numbers rather than adding
    a duplicate row, so results accumulate cleanly across sessions.
    """
    try:
        import openpyxl
    except ImportError:
        print("  [Excel export skipped: run `pip install openpyxl` to enable.]")
        return

    headers = []
    for row in rows:                                        # column order: comparison first, then metrics
        headers += [key for key in row if key not in headers]

    if output_path.exists():
        workbook = openpyxl.load_workbook(output_path)
        sheet = workbook["Results"] if "Results" in workbook.sheetnames else workbook.create_sheet("Results")
    else:
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.title = "Results"

    for column, header in enumerate(headers, start=1):
        sheet.cell(row=1, column=column, value=header)

    # Find which spreadsheet row each existing comparison is on, so it can be updated in place
    existing = {sheet.cell(row=r, column=1).value: r
                for r in range(2, sheet.max_row + 1) if sheet.cell(row=r, column=1).value}
    next_row = sheet.max_row + 1

    for row in rows:
        target = existing.get(row["comparison"], next_row)
        if target == next_row:
            next_row += 1
        for column, header in enumerate(headers, start=1):
            sheet.cell(row=target, column=column, value=row.get(header))

    workbook.save(output_path)
    print(f"\nSaved {len(rows)} result(s) to {output_path.resolve()}")


def print_summary_table(rows: list):
    """Prints all results as an aligned table."""
    if not rows:
        return
    headers = list(rows[0])
    widths = {h: max(len(h), max(len(f"{r[h]:.4f}" if isinstance(r[h], float) else str(r[h])) for r in rows))
              for h in headers}
    print("\n" + "=" * 60 + "\nSUMMARY -- all comparisons vs. reference\n" + "=" * 60)
    print(" | ".join(h.ljust(widths[h]) for h in headers))
    for row in rows:
        print(" | ".join((f"{row[h]:.4f}" if isinstance(row[h], float) else str(row[h])).ljust(widths[h])
                         for h in headers))


# ---------------------------------------------------------------------------
# 5. VOLUME SELECTION
# ---------------------------------------------------------------------------
def clean_argv(raw_args: list) -> list:
    """Drops IDE-injected flags (e.g. Spyder's --wdir, Jupyter's -f) and their values."""
    cleaned, skip = [], False
    for token in raw_args:
        if skip:
            skip = False
        elif token.startswith("-"):
            skip = True                                     # the next token is this flag's value
        else:
            cleaned.append(token)
    return cleaned


def choose_volume(label: str, allow_batch: bool = False) -> str:
    """Prompts for a volume, listing the registered names by group."""
    groups = {}
    for name, info in VOLUMES.items():
        groups.setdefault(info.get("group", "other"), []).append(name)
    print("\nAvailable volumes:")
    for group, names in groups.items():
        print(f"  [{group}] {', '.join(names)}")
    if allow_batch:
        methods = sorted({i.get("method") for i in VOLUMES.values()})
        print(f"  Batch filters: 'all', 'raw', 'denoised', or a method tag ({', '.join(methods)})")
    print("  (or paste a full file path)")

    while True:
        choice = input(f"{label} volume, filter, or path: ").strip()
        if choice:
            return choice
        print("  Please enter a volume name, a batch filter, or a file path.")


def resolve_selection() -> tuple:
    """Takes both volumes from the command line, or asks for them."""
    args = clean_argv(sys.argv[1:])
    if len(args) >= 2:
        return args[0], args[1]
    print("No volumes given. Tip: python ssim_3d.py 354ms-Optimal denoised")
    return choose_volume("Reference"), choose_volume("Comparison", allow_batch=True)


def main():
    ref_token, cmp_token = resolve_selection()
    batch = select_volumes(cmp_token, exclude=ref_token)     # None if cmp_token is a single volume

    if batch is None:
        save_results_to_excel([run_single_comparison(ref_token, cmp_token)])
        return

    print(f"\nBatch mode ('{cmp_token}'): {', '.join(batch)}")
    rows = [run_single_comparison(ref_token, name) for name in batch]
    print_summary_table(rows)
    save_results_to_excel(rows)


if __name__ == "__main__":
    main()
