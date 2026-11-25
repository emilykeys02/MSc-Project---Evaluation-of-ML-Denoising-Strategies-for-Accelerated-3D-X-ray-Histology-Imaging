"""
Train a Noise2Inverse denoiser on your K subset reconstructions, then
apply it to your full (all-projection) reconstruction.

Follows Hendriksen et al. 2020 "Noise2Inverse: Self-supervised deep
convolutional denoising for tomography":
  - X:1 training strategy (input = mean of K-1 sub-reconstructions,
    target = the remaining one)
  - Adam optimizer, learning rate 1e-3, MSE loss
  - DnCNN by default (see note on network choice below)

Epoch count and patch size follow the micro-CT-specific adaptation of
this method (Larsson, master's thesis applying N2I to microCT): since
the X:1 strategy yields K input-target pairs per slice (one per choice
of held-out subset), the effective dataset size is K times larger than
the number of slices, so epochs are reduced accordingly (25 instead of
the original paper's 100) and training uses random 512x512 patches
rather than full slices.

Prerequisites:
  - The `noise2inverse` package folder (next to this script) containing
    __init__.py, unet.py, dncnn.py, datasets.py, tiffs.py.
  - Each of your K subset reconstructions exported as a folder of
    per-slice .tif images (use vol_to_tiffs.py first if you only have
    raw .vol files from CT Pro/VGStudio).
  - A folder of per-slice .tif images for the FULL reconstruction you
    want to denoise (same conversion applies).
  - pip install torch tifffile tqdm numpy   (use a CUDA build of torch
    if you have an Nvidia GPU -- training on CPU will be very slow for
    volumes this size)

Run in Spyder: edit the CONFIG block below and press Run.
Run from PowerShell:  python train_and_denoise_noise2inverse.py
"""

import os
import sys
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import tifffile

# Make the local `noise2inverse` package folder importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from noise2inverse.datasets import Noise2InverseDataset
from noise2inverse import UNet, DnCNN


class VolDataset(Dataset):
    """Memory-maps a raw .vol file directly instead of reading thousands of
    individual per-slice TIFF files. Each physical slice gets read multiple
    times per epoch by Noise2InverseDataset (once per K-way input/target
    combination) and again every epoch -- memory-mapping lets the OS page
    cache automatically keep hot slices in RAM across these repeat reads,
    without any bespoke caching logic, and removes the per-file-open
    overhead of the TIFF-stack approach entirely."""

    def __init__(self, vol_path, shape):
        self.data = np.memmap(vol_path, dtype=np.float32, mode="r", shape=shape)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, i):
        img = np.array(self.data[i])  # copies this slice out of the mapped file
        return torch.from_numpy(img[None, ...])  # add channel dim


class SliceRangeDataset(Dataset):
    """Restricts a dataset to a contiguous range of slice indices [start, end),
    used to carve out a validation block that training never sees."""

    def __init__(self, base_ds, start, end):
        self.base_ds = base_ds
        self.start = start
        self.end = end

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, i):
        return self.base_ds[self.start + i]

# ============================= CONFIG =============================
# Paths to each of the K subset .vol reconstructions (of ONE scan, e.g. 67ms).
# Order doesn't matter. Read directly via memory-mapping -- no TIFF conversion needed.
SUBSET_VOL_PATHS = [
    r"M:\Data3\CTData\ek1e25\N2I_subsets\67ms\subset_0\proj_01\proj_01.vol",
    r"M:\Data3\CTData\ek1e25\N2I_subsets\67ms\subset_1\proj_01\proj_01.vol",
    r"M:\Data3\CTData\ek1e25\N2I_subsets\67ms\subset_2\proj_01\proj_01.vol",
    r"M:\Data3\CTData\ek1e25\N2I_subsets\67ms\subset_3\proj_01\proj_01.vol",
]
VOL_SHAPE = (2499, 1544, 2025)  # (Z, Y, X) -- confirmed against your .vgi files

# Path to the FULL (all-projection) reconstruction .vol of the SAME scan --
# this is what gets denoised at the end. Fill in once it's reconstructed.
FULL_RECON_VOL_PATH = r"M:\Data3\CTData\ek1e25\VarExposureScans\20260730_XRH_5332_OLK_DEMO02019-FFPE_67ms [2026-07-30 14.48.41]\20260730_XRH_5332_OLK_DEMO02019-FFPE_67ms\20260730_XRH_5332_OLK_DEMO02019-FFPE_67ms.vol"

# Where to save the denoised output and the trained model.
OUTPUT_DIR = r"M:\Data3\CTData\ek1e25\N2I_subsets\Denoised2\67ms\model"
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "n2i_unet_67ms.pt")

# Data range for normalization -- must be a SHARED range wide enough to cover
# Normalization range. Values are clipped to [DATA_MIN, DATA_MAX] before being
# scaled to [0,1]. This is the CLIP range, not necessarily the data's true min/max.
#
# IMPORTANT for data with a small population of very-high-value voxels
# (e.g. calcifications, metal): do NOT set DATA_MAX to the true maximum. If the
# calcifications sit near ~2700 while paraffin/tissue are down around ~150, using
# DATA_MAX=2700 squashes everything you care about into the bottom few percent of
# [0,1], the network barely trains on those low values, and it produces large dark
# overshoot ("black blob") halos around the rare bright voxels. Instead set DATA_MAX
# to roughly the 99.9th percentile of the volume (well above tissue, far below the
# calcifications). Calcifications then clip to 1.0 -- rendered flat white with no
# fine detail, which is fine since they aren't the structure of interest -- and the
# overshoot halos go away. Run a percentile check on the full reconstruction to pick
# this. Use the SAME DATA_MIN/DATA_MAX across all four exposures for comparability.
DATA_MIN = -640.242
DATA_MAX = 430          # <-- SET MANUALLY. Placeholder; replace with ~99.9th
                          # percentile of your volume (NOT the calcification max).

NETWORK = "unet"        # "unet" (highest measured PSNR in the paper's architecture
                        # comparison, Fig. 12) or "dncnn" (paper's simpler, lower-
                        # memory option -- comparable overfitting behavior to U-Net
                        # under long training, so this isn't a "safer" fallback)
PATCH_SIZE = 512         # reduced from 512 after hitting near-OOM at patch=512/batch=14 --
                         # memory scales ~quadratically with patch size, so this frees up
                         # much more headroom than shrinking batch size alone would; still
                         # over 2x your original 120px setting, so retains more spatial
                         # context for capturing correlated streak noise than that would
BATCH_SIZE = 12          # restored given the memory freed by the smaller patch size;
                         # reduce again if you still see OOM warnings at this combination
LEARNING_RATE = 1e-3

LOSS = "huber"            # "huber" = MSE for errors below HUBER_DELTA, L1 above it --
                          #   robust to the rare very-bright calcification voxels that
                          #   dominate plain MSE. NOTE: this is a deliberate deviation
                          #   from the paper, which uses "mse". "l1" and "mse" also
                          #   available.
HUBER_DELTA = 0.05        # threshold (in normalized [0,1] units) where Huber switches
                          #   from quadratic to linear. Must be well below 1.0 or Huber
                          #   collapses to plain MSE. ~0.01-0.1 is a sensible range.
                          #   Only used when LOSS = "huber".
EPOCHS = 50              # ceiling on training epochs -- early stopping (below) will
                         # typically stop before this if validation loss plateaus
NUM_WORKERS = 0          # set for a 24-logical-core CPU; leaves headroom rather
                          # than maxing out all cores for data loading

VAL_FRACTION = 0.2        # fraction of Z-slices held out for validation, never
                          # trained on -- taken as a contiguous block at the end
                          # of the volume
PATIENCE = 15             # stop if validation loss hasn't improved for this many
                          # consecutive epochs
MIN_DELTA = 1e-6          # reduced from 1e-5 after observing it was comparable in
                          # scale to actual validation losses (~1e-5 to 1e-4) --
                          # 1e-5 was masking genuine small improvements as
                          # "no improvement" (e.g. epoch 5's 0.000019 beat epoch
                          # 4's 0.000020 but didn't clear the old threshold)

TILE_INFERENCE = False    # False = denoise each full slice in one pass (no tiling,
                          # no seams). Inference under no_grad uses far less memory
                          # than training, so a full 2025x1544 slice fits on 20GB.
                          # Set True only if the denoising step hits CUDA OOM.
OVERLAP = 0.5             # tile overlap fraction, only used if TILE_INFERENCE=True.
                          # 0.25 produced a grid of seams: the U-Net's zero-padded
                          # convolutions distort each tile out past a 64px overlap.

INFERENCE_MODE = "paper"  # "paper" = Noise2Inverse Eq. 20: for each of the K subsets j,
                          #   feed the mean of the OTHER K-1 subsets (exactly the training
                          #   input distribution) and average the K network outputs. This
                          #   is the paper's method and avoids the intensity bias seen when
                          #   feeding the full reconstruction.
                          # "full" = feed the full all-projection reconstruction (the
                          #   variant tested in the microCT thesis). Out-of-distribution
                          #   vs training inputs; produced ~20% contrast shrinkage on
                          #   bright features.

OUTPUT_FORMAT = "vol"     # "vol" = one raw float32 binary (.vol) in Z order, plus a .vgi
                          #   header so CT Pro / VGStudio can open it (matches your input
                          #   .vol layout). "tiff" = one .tif per slice.
# ====================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PatchNormWrapper(Dataset):
    """Wraps a Noise2InverseDataset to normalize to [0,1]. If `deterministic`
    is False (default, used for training), takes a random crop each call so
    the network sees varied spatial context across epochs. If True (used for
    validation), always takes the same center crop, so validation loss is
    directly comparable epoch-to-epoch and reflects genuine model changes
    rather than which random patch happened to be sampled that epoch --
    this matters because validation loss determines not just when training
    stops, but which epoch's weights get saved as the final model."""

    def __init__(self, base_ds, patch_size, data_min, data_max, deterministic=False):
        self.base_ds = base_ds
        self.patch_size = patch_size
        self.lo = data_min
        self.hi = data_max
        self.deterministic = deterministic

    def __len__(self):
        return len(self.base_ds)

    def _normalize(self, x):
        return ((x - self.lo) / (self.hi - self.lo)).clamp(0.0, 1.0)

    def __getitem__(self, i):
        inp, tgt = self.base_ds[i]
        inp = self._normalize(inp.float())
        tgt = self._normalize(tgt.float())

        _, H, W = inp.shape
        p = self.patch_size
        if H > p and W > p:
            if self.deterministic:
                top = (H - p) // 2
                left = (W - p) // 2
            else:
                top = np.random.randint(0, H - p)
                left = np.random.randint(0, W - p)
            inp = inp[:, top:top + p, left:left + p]
            tgt = tgt[:, top:top + p, left:left + p]
        return inp, tgt


def build_model(name):
    if name == "dncnn":
        return DnCNN(channels=1, num_of_layers=20)
    elif name == "unet":
        return UNet(n_channels=1, n_classes=1)
    else:
        raise ValueError(f"Unknown NETWORK '{name}', expected 'dncnn' or 'unet'.")


def train():
    print(f"Using device: {DEVICE}")

    print("\nLoading subset reconstructions (memory-mapped):")
    for p in SUBSET_VOL_PATHS:
        print(f"  {p}")
    sub_datasets = [VolDataset(p, VOL_SHAPE) for p in SUBSET_VOL_PATHS]
    n_slices = len(sub_datasets[0])
    for p, ds in zip(SUBSET_VOL_PATHS, sub_datasets):
        if len(ds) != n_slices:
            raise RuntimeError(
                f"Subset volumes have mismatched slice counts: {p} has {len(ds)}, "
                f"expected {n_slices}. All K subsets must have identical Z-extent."
            )
    print(f"Found {n_slices} slices per subset, {len(sub_datasets)} subsets.")

    n_val = max(1, int(n_slices * VAL_FRACTION))
    n_train = n_slices - n_val
    print(f"Splitting: {n_train} slices for training, {n_val} held out for validation "
          f"(slices {n_train}-{n_slices-1}).")

    train_sub_datasets = [SliceRangeDataset(ds, 0, n_train) for ds in sub_datasets]
    val_sub_datasets = [SliceRangeDataset(ds, n_train, n_slices) for ds in sub_datasets]

    n2i_train = Noise2InverseDataset(*train_sub_datasets, strategy="X:1")
    n2i_val = Noise2InverseDataset(*val_sub_datasets, strategy="X:1")
    print(f"Noise2Inverse X:1 training pairs: {len(n2i_train)}, validation pairs: {len(n2i_val)}")

    train_ds = PatchNormWrapper(n2i_train, PATCH_SIZE, DATA_MIN, DATA_MAX, deterministic=False)
    val_ds = PatchNormWrapper(n2i_val, PATCH_SIZE, DATA_MIN, DATA_MAX, deterministic=True)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, drop_last=False
    )

    model = build_model(NETWORK).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    if LOSS == "huber":
        loss_fn = nn.HuberLoss(delta=HUBER_DELTA)
        print(f"Loss: Huber (delta={HUBER_DELTA})")
    elif LOSS == "l1":
        loss_fn = nn.L1Loss()
        print("Loss: L1")
    elif LOSS == "mse":
        loss_fn = nn.MSELoss()
        print("Loss: MSE (paper default)")
    else:
        raise ValueError(f"Unknown LOSS '{LOSS}', expected 'huber', 'l1', or 'mse'.")

    print(f"\nTraining {NETWORK} for up to {EPOCHS} epochs "
          f"(early stopping: patience={PATIENCE}, min_delta={MIN_DELTA})...")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    training_start = time.time()

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    stopped_early = False

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        for inp, tgt in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [train]"):
            inp, tgt = inp.to(DEVICE), tgt.to(DEVICE)
            optimizer.zero_grad()
            pred = model(inp)
            loss = loss_fn(pred, tgt)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inp.size(0)
        avg_train_loss = running_loss / len(train_ds)

        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for inp, tgt in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [val]", leave=False):
                inp, tgt = inp.to(DEVICE), tgt.to(DEVICE)
                pred = model(inp)
                loss = loss_fn(pred, tgt)
                val_running_loss += loss.item() * inp.size(0)
        avg_val_loss = val_running_loss / len(val_ds)

        epoch_duration = time.time() - epoch_start
        elapsed = time.time() - training_start
        avg_epoch_time = elapsed / (epoch + 1)
        remaining = avg_epoch_time * (EPOCHS - epoch - 1)

        improved = avg_val_loss < (best_val_loss - MIN_DELTA)
        if improved:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            improvement_note = "(best so far, saved)"
        else:
            epochs_without_improvement += 1
            improvement_note = f"(no improvement, {epochs_without_improvement}/{PATIENCE})"

        print(f"Epoch {epoch+1}/{EPOCHS} -- train MSE: {avg_train_loss:.6f} -- "
              f"val MSE: {avg_val_loss:.6f} {improvement_note} -- "
              f"took {epoch_duration/60:.1f} min -- elapsed {elapsed/60:.1f} min -- "
              f"est. remaining {remaining/60:.1f} min")

        if epochs_without_improvement >= PATIENCE:
            print(f"\nEarly stopping: no validation improvement for {PATIENCE} epochs. "
                  f"Stopping after epoch {epoch+1}/{EPOCHS}.")
            stopped_early = True
            break

    total_time = time.time() - training_start
    print(f"\nTraining finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total training time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    if stopped_early:
        print(f"Stopped early; using best weights from validation loss {best_val_loss:.6f}.")
    elif best_state is not None:
        print(f"Completed all {EPOCHS} epochs; using best weights from validation loss "
              f"{best_val_loss:.6f} (may not be the final epoch).")

    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Saved trained model to {MODEL_SAVE_PATH}")
    return model


def sliding_window_denoise(model, img, patch_size=512, stride=None):
    """Run a fully-convolutional model over a large image in overlapping
    tiles, blending overlaps with a smooth (Hann) window rather than a
    naive uniform average. A naive average creates visible seams at tile
    boundaries because edge-of-tile predictions are less reliable than
    tile-center predictions; weighting by a window peaked at each tile's
    center and tapering toward its edges removes those seams."""
    if stride is None:
        stride = max(1, int(patch_size * (1.0 - OVERLAP)))
    H, W = img.shape[-2:]
    out = torch.zeros_like(img)
    weight = torch.zeros_like(img)

    w1d = np.hanning(patch_size)
    w1d = np.clip(w1d, 0.1, None)  # floor so tile edges still contribute something
    window_np = np.outer(w1d, w1d).astype(np.float32)
    window = torch.from_numpy(window_np).to(img.device)

    top_positions = list(range(0, max(H - patch_size, 0) + 1, stride))
    left_positions = list(range(0, max(W - patch_size, 0) + 1, stride))
    if top_positions[-1] != H - patch_size:
        top_positions.append(max(H - patch_size, 0))
    if left_positions[-1] != W - patch_size:
        left_positions.append(max(W - patch_size, 0))

    with torch.no_grad():
        for top in top_positions:
            for left in left_positions:
                patch = img[..., top:top + patch_size, left:left + patch_size]
                pred = model(patch.unsqueeze(0)).squeeze(0)
                out[..., top:top + patch_size, left:left + patch_size] += pred * window
                weight[..., top:top + patch_size, left:left + patch_size] += window
    return out / weight.clamp(min=1e-6)


def _run_model(model, img_norm):
    if TILE_INFERENCE:
        return sliding_window_denoise(model, img_norm, patch_size=PATCH_SIZE)
    with torch.no_grad():
        return model(img_norm.unsqueeze(0)).squeeze(0)


def _normalize(img):
    return (((img.float() - DATA_MIN) / (DATA_MAX - DATA_MIN)).clamp(0.0, 1.0)).to(DEVICE)


class SliceWriter:
    """Writes denoised slices either as a single raw float32 .vol (streamed in
    Z order, so memory stays flat) plus a matching .vgi header, or as one .tif
    per slice. Tracks the running min/max of the written data for the .vgi."""

    def __init__(self, output_dir, fmt, n_slices, shape):
        self.fmt = fmt
        self.output_dir = output_dir
        self.n_slices = n_slices
        self.shape = shape          # (Z, Y, X)
        self.vmin = np.inf
        self.vmax = -np.inf
        if fmt == "vol":
            self.vol_path = os.path.join(output_dir, "denoised.vol")
            self.fh = open(self.vol_path, "wb")
        elif fmt == "tiff":
            self.fh = None
        else:
            raise ValueError(f"Unknown OUTPUT_FORMAT '{fmt}', expected 'vol' or 'tiff'.")

    def write(self, i, slice2d):
        slice2d = slice2d.astype(np.float32)
        self.vmin = min(self.vmin, float(slice2d.min()))
        self.vmax = max(self.vmax, float(slice2d.max()))
        if self.fmt == "vol":
            self.fh.write(slice2d.tobytes())
        else:
            tifffile.imwrite(os.path.join(self.output_dir, f"denoised_{i:05d}.tif"), slice2d)

    def close(self):
        if self.fmt == "vol":
            self.fh.close()
            self._write_vgi()
            return self.vol_path
        return self.output_dir

    def _write_vgi(self):
        z, y, x = self.shape
        vgi = (
            "{volume1}\n"
            "[representation]\n"
            f"size = {x} {y} {z}\n"
            "datatype = float\n"
            f"datarange = {self.vmin} {self.vmax}\n"
            "bitsperelement = 32\n"
            "[file1]\n"
            f"RegionOfInterestStart = 0 0 0\n"
            f"RegionOfInterestEnd = {x-1} {y-1} {z-1}\n"
            "FileFormat = raw\n"
            f"Size = {x} {y} {z}\n"
            "Name = denoised.vol\n"
            "Datatype = float\n"
            f"datarange = {self.vmin} {self.vmax}\n"
            "BitsPerElement = 32\n"
            "{volumeprimitive1}\n"
            "[geometry]\n"
            "resolution = 0.0163692 0.0163692 0.0163692\n"
            "unit = mm\n"
            "[description]\n"
            "text = denoised\n"
        )
        with open(os.path.join(self.output_dir, "denoised.vgi"), "w") as f:
            f.write(vgi)


def denoise_full_reconstruction(model):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.eval()

    if INFERENCE_MODE == "paper":
        print(f"\nDenoising (paper mode, Eq. 20 -- averaging over K sub-reconstruction inputs)")
        sub_ds = [VolDataset(p, VOL_SHAPE) for p in SUBSET_VOL_PATHS]
        K = len(sub_ds)
        n_slices = len(sub_ds[0])
        print(f"Found {n_slices} slices. Writing {OUTPUT_FORMAT}.")
        writer = SliceWriter(OUTPUT_DIR, OUTPUT_FORMAT, n_slices, VOL_SHAPE)

        for i in tqdm(range(n_slices), desc="Denoising slices"):
            subs = [_normalize(ds[i]) for ds in sub_ds]  # K x (1,H,W)
            acc = None
            for j in range(K):
                inp = torch.stack([subs[k] for k in range(K) if k != j]).mean(dim=0)
                out = _run_model(model, inp)
                acc = out if acc is None else acc + out
            denoised_norm = acc / K
            denoised = denoised_norm.cpu().numpy() * (DATA_MAX - DATA_MIN) + DATA_MIN
            writer.write(i, denoised[0])
        result = writer.close()

    elif INFERENCE_MODE == "full":
        print(f"\nDenoising (full mode) from:\n  {FULL_RECON_VOL_PATH}")
        full_ds = VolDataset(FULL_RECON_VOL_PATH, VOL_SHAPE)
        n_slices = len(full_ds)
        print(f"Found {n_slices} slices to denoise. Writing {OUTPUT_FORMAT}.")
        writer = SliceWriter(OUTPUT_DIR, OUTPUT_FORMAT, n_slices, VOL_SHAPE)

        for i in tqdm(range(n_slices), desc="Denoising slices"):
            denoised_norm = _run_model(model, _normalize(full_ds[i]))
            denoised = denoised_norm.cpu().numpy() * (DATA_MAX - DATA_MIN) + DATA_MIN
            writer.write(i, denoised[0])
        result = writer.close()
    else:
        raise ValueError(f"Unknown INFERENCE_MODE '{INFERENCE_MODE}', expected 'paper' or 'full'.")

    print(f"\nDone. Denoised output written to:\n  {result}")


if __name__ == "__main__":
    trained_model = train()
    denoise_full_reconstruction(trained_model)
