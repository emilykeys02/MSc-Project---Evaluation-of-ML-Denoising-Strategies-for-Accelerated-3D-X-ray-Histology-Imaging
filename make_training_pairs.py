"""
Build clean/noisy training pairs for the supervised projection-domain models.

Dragonfly expects the clean (reference) and noisy projections as two folders of
matching files, so this script copies the chosen projections out of the
linearised scan folders and renames them with a shared pair index:

    <OUT_DIR>/clean/clean_pair_0001.tif   <-  reference projection
    <OUT_DIR>/67ms/67ms_pair_0001.tif     <-  the 67ms projection at the same angle
    ...and the same for the other exposures.

Pairs are matched by rotation angle, not by index: the reference scan has 3502
projections and the lower-exposure scans have 2002, so the two index sequences
advance at different rates. Taking every 35th reference projection and every
20th lower-exposure projection keeps both on the same angles, giving 100 pairs
evenly spaced every 3.6 degrees around the full rotation. The largest angular
mismatch across all 100 pairs is 0.0118 degrees.

All four lower-exposure scans share the same 2002-projection angle table, so
one index list works for all of them; only the reference side differs.

Run with Python 3 on the machine where the network drive is mounted.
"""

import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. SOURCE FOLDERS
#    Every scan is a folder of linearised projections named
#    <prefix><index>.tif with a 4-digit zero-padded index, e.g.
#    20260730_XRH_5332_OLK_DEMO02019-FFPE_67ms_0001.tif
# ---------------------------------------------------------------------------
LINEAR_DIR = Path(r"M:\Data3\CTData\ek1e25\Real scan projections linearised\Linearised Projections")
SAMPLE_ID = "20260730_XRH_5332_OLK_DEMO02019-FFPE"

# Reference: 354ms, 4-frame averaged ("Optimal"), 3502 projections
CLEAN_SRC_DIR = LINEAR_DIR / "Rea354ms_optimalprojections_linear"
CLEAN_PREFIX = f"{SAMPLE_ID}_354-Optimal_"

# Lower-exposure scans: label -> (source folder, filename prefix).
# Folder names are as they exist on disk, including the typo in the 354ms one.
EXPOSURES = {
    "177ms": (LINEAR_DIR / "Real177msprojections_linear", f"{SAMPLE_ID}_177ms_"),
    "354ms": (LINEAR_DIR / "Rea354msmsprojections_linear", f"{SAMPLE_ID}_354ms_"),
    "89ms":  (LINEAR_DIR / "Real89msprojections_linear", f"{SAMPLE_ID}_89ms_"),
    "67ms":  (LINEAR_DIR / "Real67msprojections_linear", f"{SAMPLE_ID}_67ms_"),
}

OUT_DIR = Path(r"M:\Data3\CTData\ek1e25\training_pairs_100")
SRC_PAD = OUT_PAD = 4                  # zero-padding of the index in input and output names
SRC_EXT = ".tif"

# ---------------------------------------------------------------------------
# 2. MATCHED PAIRS (clean index, noisy index)
#    Both sequences start at projection 1 and step through the rotation at the
#    rate that keeps them on the same angles: 3502/100 ~ 35 for the reference,
#    2002/100 ~ 20 for the lower-exposure scans.
# ---------------------------------------------------------------------------
N_PAIRS = 100
CLEAN_STEP, NOISY_STEP = 35, 20
PAIRS = [(1 + CLEAN_STEP * k, 1 + NOISY_STEP * k) for k in range(N_PAIRS)]


# ---------------------------------------------------------------------------
# 3. COPYING
# ---------------------------------------------------------------------------
def src_filename(prefix: str, index: int) -> str:
    """Source name, e.g. <prefix>0141.tif"""
    return f"{prefix}{index:0{SRC_PAD}d}{SRC_EXT}"


def out_filename(pair_num: int, label: str) -> str:
    """Destination name, e.g. clean_pair_0008.tif"""
    return f"{label}_pair_{pair_num:0{OUT_PAD}d}{SRC_EXT}"


def copy_one(src_path: Path, dst_path: Path, missing: list) -> bool:
    """Copies one projection, recording rather than raising on a missing file."""
    if not src_path.exists():
        missing.append(str(src_path))                      # collect them all, report at the end
        return False
    shutil.copy2(src_path, dst_path)                       # copy2 keeps the timestamps
    return True


def main():
    clean_out = OUT_DIR / "clean"
    clean_out.mkdir(parents=True, exist_ok=True)
    exposure_out = {name: OUT_DIR / name for name in EXPOSURES}
    for folder in exposure_out.values():
        folder.mkdir(parents=True, exist_ok=True)

    missing = []
    copied = {"clean": 0, **{name: 0 for name in EXPOSURES}}

    for pair_num, (clean_idx, noisy_idx) in enumerate(PAIRS, start=1):
        # reference projection for this pair
        if copy_one(CLEAN_SRC_DIR / src_filename(CLEAN_PREFIX, clean_idx),
                    clean_out / out_filename(pair_num, "clean"), missing):
            copied["clean"] += 1

        # the same angle from each lower-exposure scan
        for name, (src_dir, prefix) in EXPOSURES.items():
            if copy_one(src_dir / src_filename(prefix, noisy_idx),
                        exposure_out[name] / out_filename(pair_num, name), missing):
                copied[name] += 1

        print(f"[{pair_num:03d}/{len(PAIRS)}] clean {clean_idx:04d} <-> noisy {noisy_idx:04d}")

    print("\nDone.")
    for name, count in copied.items():
        print(f"  {name}: {count}/{len(PAIRS)} copied")
    if missing:
        print(f"\n{len(missing)} file(s) not found (check the paths and padding):")
        for path in missing:
            print("  -", path)


if __name__ == "__main__":
    main()
