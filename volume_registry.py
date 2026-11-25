"""
Shared volume registry for the image-quality scripts (PSNR, SSIM).

Every reconstructed volume used in this study is a headerless (or
header-prefixed) binary file of float32 voxels with the same dimensions, so
this module holds one place that:

  * records the volume dimensions and data type,
  * lists where each scan / denoised output lives on the network drive,
  * resolves a short name (e.g. "unetpp-huber-67ms") to a file path.

Most denoised outputs follow a regular folder/file naming pattern, so paths
are built from one template per method and only the handful of files that
were named differently are listed individually in PATH_OVERRIDES. This keeps
the registry short rather than repeating ~30 near-identical full paths.

Volume naming convention:  <method>-<exposure>   e.g. "generic-89ms"
Raw (undenoised) scans are named by exposure alone, e.g. "67ms".
"""

from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# 1. VOLUME GEOMETRY (identical for every file in the study)
# ---------------------------------------------------------------------------
NX, NY, NZ = 2025, 1544, 2499      # voxels: "size = 2025 1544 2499" from the .vgi header
DTYPE_LE = np.dtype("<f4")         # little-endian float32: the normal case
DTYPE_BE = np.dtype(">f4")         # big-endian float32: only the Med+Gaus outputs
CANDIDATE_EXTENSIONS = [".vol", ".raw"]   # tried in this order when resolving a raw scan

# ---------------------------------------------------------------------------
# 2. WHERE THE DATA LIVES
# ---------------------------------------------------------------------------
BASE_DIR = Path(r"M:\Data3\CTData\ek1e25\VarExposureScans")   # the five original scans
DENOISED_BASE_DIR = Path(r"M:\Data3\CTData\ek1e25")           # all denoised outputs

EXPOSURES = ["67ms", "89ms", "177ms", "354ms"]   # the four exposures every method was applied to
SAMPLE_ID = "20260730_XRH_5332_OLK_DEMO02019-FFPE"

# ---------------------------------------------------------------------------
# 3. RAW SCANS -- name -> (timestamped folder suffix, file tag, datarange)
#    Folder name is "<SAMPLE_ID>_<tag> [<timestamp>]"; datarange is recorded
#    for reference only (the scripts use one shared range for all comparisons).
# ---------------------------------------------------------------------------
RAW_SCANS = {
    "67ms":          ("67ms",         "2026-07-30 14.48.41", (-412.897, 2657.29)),
    "89ms":          ("89ms",         "2026-07-30 14.55.44", (-415.296, 2642.48)),
    "177ms":         ("177ms",        "2026-07-30 15.05.43", (-377.783, 2658.43)),
    "354ms":         ("354ms",        "2026-07-30 15.24.31", (-355.227, 2679.42)),
    "354ms-Optimal": ("354-Optimal",  "2026-07-30 15.48.29", (-347.171, 2683.96)),
}

REFERENCE_KEY = "354ms-Optimal"   # the clean reference every other volume is compared against

# ---------------------------------------------------------------------------
# 4. DENOISED OUTPUTS
#    One template per method, with {exp} replaced by the exposure label.
#    Example of a fully expanded path (generic, 89ms):
#      M:\Data3\CTData\ek1e25\Real Scan modelling stuff\Modelling on
#      reconstructions\Generic Denoising\Model applied denormalised Final\89\
#      89_denormalised_scan.vol
# ---------------------------------------------------------------------------
RECON_DIR = r"Real Scan modelling stuff\Modelling on reconstructions"   # denoised after reconstruction
PROJ_DIR = r"Real Scan modelling stuff\Modelling on projections"        # denoised on projections

METHOD_TEMPLATES = {
    # method tag: (path template, byteswap?)
    "medgaus":   (RECON_DIR + r"\Med+Gaus Filtering\{exp}\21110_" + SAMPLE_ID + r"_{exp}.raw", True),
    "generic":   (RECON_DIR + r"\Generic Denoising\Model applied denormalised Final\{num}\{num}_denormalised_scan.vol", False),
    "n2v-recon": (RECON_DIR + r"\N2V 0308\{exp}\Inference N2V Model RealScan {exp} on reconstruction.raw", False),
    "n2v-proj":  r"RECONSTRUCTED\N2V 200\{exp}\Inference_Noise2Void_{exp}_200\Inference_Noise2Void_{exp}_200.vol",
    "100pairclean-huber":     r"TO_BE_RECONSTRUCTED\100pairsclean huberloss\{exp}\Inference_unet_100clean_huber_{exp}_01\Inference_unet_100clean_huber_{exp}_01.vol",
    "100pairclean-mixedgrad": PROJ_DIR + r"\mixed gradients\linearised\{exp}\Inference_Unet_mixedgrad_{exp}_01\Inference_Unet_mixedgrad_{exp}_01.vol",
    "unetpp-huber":           r"TO_BE_RECONSTRUCTED\Unet++ huberloss\{exp}\Inference_unetplus_huber_{exp}_01\Inference_unetplus_huber_{exp}_01.vol",
}

# Files that were saved under a different folder or file name than the template
# above would produce. Listed individually so the template stays simple.
PATH_OVERRIDES = {
    ("medgaus", "89ms"):   RECON_DIR + r"\Med+Gaus Filtering\89\21110_" + SAMPLE_ID + r"_89ms.raw",
    ("medgaus", "177ms"):  RECON_DIR + r"\Med+Gaus Filtering\177ms redone\21110_" + SAMPLE_ID + r"_177ms.raw",
    ("medgaus", "354ms"):  RECON_DIR + r"\Med+Gaus Filtering\354\21110_" + SAMPLE_ID + r"_354ms.raw",
    ("generic", "354ms"):  RECON_DIR + r"\Generic Denoising\Model applied denormalised Final\354\denormalised_scan_354.vol",
    ("n2v-recon", "89ms"): RECON_DIR + r"\N2V 0308\89\Inference N2V Model RealScan 88p5ms on reconstruction.raw",
    # NB: the 177ms file really does have a space before ".raw" -- kept exactly as on disk
    ("n2v-recon", "177ms"): RECON_DIR + r"\N2V 0308\177\Inference N2V Model RealScan 177ms on reconstruction .raw",
    ("n2v-recon", "354ms"): RECON_DIR + r"\N2V 0308\354ms new\Inference N2V Model354ms redone.raw",
    ("n2v-proj", "354ms"):  r"RECONSTRUCTED\N2V 200\354ms\Inference_Unet_Noise2Void_354ms_01\Inference_Unet_Noise2Void_354ms_01.vol",
    ("unetpp-huber", "67ms"):  r"TO_BE_RECONSTRUCTED\Unet++ huberloss\67ms\Inference_unetpp_67ms_01\Inference_unetpp_67ms_01.vol",
    ("unetpp-huber", "354ms"): r"TO_BE_RECONSTRUCTED\Unet++ huberloss\354ms\Inference_unet_100clean_huber_354ms_01\Inference_unet_100clean_unet++_354ms_01.vol",
    # Noise2Inverse outputs go here once reconstructed.
}


def build_volumes() -> dict:
    """Builds the full {name: info} registry of raw scans and denoised outputs."""
    volumes = {}

    for name, (tag, timestamp, datarange) in RAW_SCANS.items():
        volumes[name] = {                                  # raw scans are found by folder + basename
            "subfolder": f"{SAMPLE_ID}_{tag} [{timestamp}]",
            "basename": f"{SAMPLE_ID}_{tag}",
            "datarange": datarange, "group": "raw", "method": "raw",
        }

    for method, template in METHOD_TEMPLATES.items():
        template, byteswap = template if isinstance(template, tuple) else (template, False)
        for exp in EXPOSURES:
            relative = PATH_OVERRIDES.get(
                (method, exp),                             # use the override if this file was named differently
                template.format(exp=exp, num=exp.replace("ms", "")),   # {num} = exposure without "ms" (Generic folders)
            )
            volumes[f"{method}-{exp}"] = {
                "full_path": DENOISED_BASE_DIR / relative,
                "group": "denoised", "method": method,
            }
            if byteswap:
                volumes[f"{method}-{exp}"]["byteswap"] = True   # Med+Gaus files are big-endian
    return volumes


VOLUMES = build_volumes()


def get_dtype(info: dict) -> np.dtype:
    """Big-endian float32 for byte-swapped files, little-endian otherwise."""
    return DTYPE_BE if info.get("byteswap") else DTYPE_LE


def resolve_path(key: str, info: dict) -> Path:
    """Returns the file path for a registry entry, raising a helpful error if missing.

    Denoised entries carry a full path; raw scans are searched for by basename
    because CT Pro sometimes writes the volume directly into the scan folder
    and sometimes into a nested folder of the same name.
    """
    if "full_path" in info:                                  # denoised outputs: path already known
        path = Path(info["full_path"])
        if not path.exists():
            raise FileNotFoundError(f"[{key}] file not found: {path}")
        return path

    folder = BASE_DIR / info["subfolder"]
    nested = folder / info["basename"]                       # <folder>/<basename>/<basename>.vol layout
    candidates = [folder / (info["basename"] + ext) for ext in CANDIDATE_EXTENSIONS]
    candidates += [nested / (info["basename"] + ext) for ext in CANDIDATE_EXTENSIONS]
    if nested.is_dir():
        for ext in CANDIDATE_EXTENSIONS:
            candidates.extend(sorted(nested.glob(f"*{ext}")))

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    # Deliberately no loose "any .vol in the folder" fallback: with several
    # scans' outputs nearby, that could silently pick the wrong file. Instead,
    # list what is actually there so the mismatch is obvious.
    lines = [f"[{key}] no file found for basename '{info['basename']}' under {folder}"]
    for directory in (folder, nested):
        if directory.exists() and directory.is_dir():
            lines.append(f"  Contents of {directory}:")
            lines += [f"    {'[DIR] ' if item.is_dir() else '      '}{item.name}"
                      for item in sorted(directory.iterdir())]
    raise FileNotFoundError("\n".join(lines))


def select_volumes(token: str, exclude: str = None) -> list:
    """Expands a batch token into a list of volume names.

    Accepts "all", "raw", "denoised", a method tag (e.g. "n2v-proj"), or a
    comma-separated combination. Returns None if the token isn't recognised,
    so the caller can treat it as a single volume name or file path instead.
    """
    methods = {info.get("method") for info in VOLUMES.values()}
    selected = []
    for part in [p.strip().lower() for p in token.split(",") if p.strip()]:
        if part == "all":
            matches = [n for n in VOLUMES if n != exclude]
        elif part in ("raw", "denoised"):
            matches = [n for n, i in VOLUMES.items() if i.get("group") == part and n != exclude]
        elif part in methods:
            matches = [n for n, i in VOLUMES.items() if i.get("method") == part and n != exclude]
        else:
            return None                                       # not a batch filter
        selected += [n for n in matches if n not in selected]  # keep order, drop duplicates
    return selected
