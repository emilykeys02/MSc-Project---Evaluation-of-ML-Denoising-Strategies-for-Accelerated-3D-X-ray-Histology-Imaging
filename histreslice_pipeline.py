"""
Reorient a reconstructed volume so that a slice through it matches the plane
of a conventional histology section.

The sample is mounted at an angle in the scanner, deliberately, so that no flat
surface of the cassette is parallel to the detector (parallel surfaces cause
reconstruction artefacts). A slice straight through the reconstructed volume is
therefore oblique to the tissue, and the volume has to be rotated back before
it can be compared with a histology section.

This is a Python translation of the facility's HistRelevantReslice ImageJ
macro, with the two rotation angles fixed rather than drawn by hand each time,
so every exposure and denoised output is sliced identically.

Steps, in the order the macro applies them:
  1. rotate about Z          in-plane rotation of every slice, canvas grows
  2. simple_crop_volume      pad, threshold one slice, crop to its bounding box
  3. reslice                 transpose, no interpolation
  4. rotate about Y          in-plane rotation of the resliced stack
  5. rotate 90 degrees right
  6. crop_to_rect            crop to the wax block (the one manual measurement)
  7. final reslice           transpose back
  8. optional stack reverse

Used by run_histology_slice.py, which supplies the angles and crop boxes.
"""

import numpy as np
from scipy.ndimage import rotate as ndi_rotate


def load_raw_volume(path, width, height, slices, dtype=np.float32, memmap=True):
    """Opens a headerless binary volume as (Z, Y, X).

    memmap=True does not read the file: slices are fetched only when indexed.
    Follow it with precrop_window_volume so that only the region of interest is
    ever pulled into memory.
    """
    if memmap:
        return np.memmap(path, dtype=dtype, mode="r", shape=(slices, height, width))

    volume = np.fromfile(path, dtype=dtype)
    expected = width * height * slices
    if volume.size != expected:
        raise ValueError(f"File holds {volume.size} voxels, expected {expected} "
                         f"({width}x{height}x{slices}). Check the dimensions and dtype.")
    return volume.reshape((slices, height, width))


def precrop_window_volume(vol, z_lo, z_hi, x_lo, x_hi, y_lo, y_hi):
    """Crops a rough box around the sample before any rotation.

    This is the macro's manual "window volume" step. np.array() forces the
    memory-mapped region to be read now, so everything downstream works on a
    contiguous array of just this sub-volume.
    """
    return np.array(vol[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi], dtype=np.float32)


def rotate_inplane_stack(vol, angle_deg, enlarge=True, order=1):
    """Rotates every slice in-plane, as ImageJ's 'Rotate... enlarge stack' does.

    axes=(1, 2) rotates within each slice, leaving the slice order untouched.
    reshape=enlarge grows the canvas so corners are not cut off.
    order=1 is bilinear, for speed; the macro used bicubic, so set order=3 for
    a closer match at the cost of runtime.
    """
    return ndi_rotate(vol, angle=angle_deg, axes=(1, 2), reshape=enlarge,
                      order=order, mode="constant", cval=0.0, prefilter=(order > 1))


def simple_crop_volume(vol, pad=25, is_32bit=True, ref_slice=None):
    """Pads the canvas, then crops every slice to the sample's bounding box.

    The bounding box is found by thresholding one representative slice (the
    middle one by default), matching the macro, which used whichever slice was
    active. Rotation leaves zero-filled corners, and this removes them.
    """
    z, y, x = vol.shape
    padded = np.zeros((z, y + pad, x + pad), dtype=vol.dtype)
    padded[:, pad // 2:pad // 2 + y, pad // 2:pad // 2 + x] = vol   # centre the volume in the padded canvas

    ref_slice = padded.shape[0] // 2 if ref_slice is None else ref_slice
    threshold = 0.01 if is_32bit else 1              # float volumes sit near 0 in air; integer ones at 0
    mask = padded[ref_slice] > threshold
    if not mask.any():
        raise ValueError("Threshold produced an empty mask -- check is_32bit and the threshold.")

    ys, xs = np.where(mask)
    y_lo, y_hi = ys.min(), ys.max() + 1              # +1 so the last row/column is included
    x_lo, x_hi = xs.min(), xs.max() + 1
    crop_box = dict(pad=pad, y_lo=y_lo, y_hi=y_hi, x_lo=x_lo, x_hi=x_hi)
    return padded[:, y_lo:y_hi, x_lo:x_hi], crop_box


def reslice_transpose(vol):
    """Re-slices the stack along a different axis: (Z, Y, X) -> (Y, Z, X).

    A pure re-indexing, like ImageJ's 'Reslice [/]... start=Top', so no
    interpolation and no loss of detail.
    """
    return np.ascontiguousarray(np.transpose(vol, (1, 0, 2)))


def rotate90_right(vol, clockwise=True):
    """Rotates every slice by 90 degrees, as 'Rotate 90 Degrees Right' does.

    If the result is 90 degrees out, try clockwise=False.
    """
    return np.ascontiguousarray(np.rot90(vol, k=-1 if clockwise else 1, axes=(1, 2)))


def crop_to_rect(vol, x, y, w, h):
    """Crops every slice to a fixed rectangle (the wax block).

    The rectangle is clipped to the volume, so a generous margin cannot push it
    out of bounds; no overlap at all is an error, since that means the
    rectangle was measured in a different orientation.
    """
    _, vy, vx = vol.shape
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(vx, x + w), min(vy, y + h)
    if x0 >= x1 or y0 >= y1:
        raise ValueError(f"Crop rectangle ({x},{y},{w},{h}) does not overlap the volume "
                         f"(Y={vy}, X={vx}).")
    return vol[:, y0:y1, x0:x1]


def process_volume(raw_path, width, height, slices,
                   rotate_about_z, rotate_about_y,
                   wax_crop_rect,            # (x, y, w, h), measured after the Y rotation and 90-degree turn
                   window_volume=None,       # (z_lo, z_hi, x_lo, x_hi, y_lo, y_hi) in raw volume coordinates
                   is_32bit=True, pad=25,
                   reverse_stack=False,
                   rotate_order=1,
                   rotate90_clockwise=True,
                   final_reslice=True):
    """Runs the full pipeline and returns (volume, first crop box)."""
    vol = load_raw_volume(raw_path, width, height, slices)

    if window_volume is not None:
        vol = precrop_window_volume(vol, *window_volume)          # read only the sample region

    vol = rotate_inplane_stack(vol, rotate_about_z, enlarge=True, order=rotate_order)   # step 1
    vol, crop_box = simple_crop_volume(vol, pad=pad, is_32bit=is_32bit)                 # step 2
    vol = reslice_transpose(vol)                                                        # step 3
    vol = rotate_inplane_stack(vol, rotate_about_y, enlarge=True, order=rotate_order)   # step 4
    vol = rotate90_right(vol, clockwise=rotate90_clockwise)                             # step 5
    vol = crop_to_rect(vol, *wax_crop_rect)                                             # step 6

    if final_reslice:
        vol = reslice_transpose(vol)                                                    # step 7
    if reverse_stack:
        vol = vol[::-1]                                                                 # step 8
    return vol, crop_box
