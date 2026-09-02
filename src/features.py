"""Part C: turning a 450-vector into something a model can consume.

Three representations are implemented, because the assignment asks for the
choice to be justified against an alternative rather than asserted:

  normalised_cutouts  -- the one this study commits to. A fixed PHYSICAL width
                         window, resampled to a fixed number of points, with
                         range measured relative to the centre beam. An object
                         at 2 m and the same object at 6 m produce the same
                         vector.
  raw_window_cutouts  -- identical in every respect except that the window is a
                         fixed number of BEAMS. This is the control that
                         isolates the angular scaling, and nothing else.
  arras_features      -- the classical 1-D geometric descriptor set (Arras,
                         Mozos & Burgard, ICRA 2007): width, circularity,
                         linearity, boundary regularity and so on, computed in
                         metres and therefore already scale-aware.

Everything here is batched over candidate beams. The inner loop is numpy over
(K, S) arrays; the only Python-level loop in the pipeline is over frames, where
each iteration already does a few hundred thousand flops.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src import labels as L  # noqa: E402

_ANG_RES_RAD = np.radians(config.ANGULAR_RES_DEG)


# ---------------------------------------------------------------------------
# Validity and candidate selection
# ---------------------------------------------------------------------------
def missing_mask(R: np.ndarray, sentinels) -> np.ndarray:
    """True where a reading is a missing-return sentinel.

    Membership in an exact set of values, never `R > 29`. The audit in Part A
    finds genuine returns above 29 m, so a threshold filter would delete real
    data and keep some sentinels depending on the file.
    """
    R = np.asarray(R, dtype=float)
    out = np.zeros(R.shape, dtype=bool)
    for s in np.atleast_1d(np.asarray(sentinels, dtype=float)):
        out |= np.isclose(R, s, atol=5e-3)
    return out


def candidate_mask(
    R: np.ndarray,
    missing: np.ndarray,
    min_r: float = config.CANDIDATE_MIN_RANGE_M,
    max_r: float = config.CANDIDATE_MAX_RANGE_M,
) -> np.ndarray:
    """Beams eligible to be a detection centre.

    Excludes sentinels, readings inside the platform footprint, and readings
    beyond the range where a person is wider than a single beam. This is a
    stated restriction of the detector's operating envelope, so any object
    outside it is counted as a miss during evaluation rather than quietly
    dropped from the ground truth.
    """
    return (~missing) & (R >= min_r) & (R <= max_r)


# ---------------------------------------------------------------------------
# The distance-normalised cutout
# ---------------------------------------------------------------------------
def cutout_half_beams(
    r0: np.ndarray,
    window_m: float = config.CUTOUT_WIDTH_M,
    min_half: float = config.CUTOUT_MIN_HALF_BEAMS,
    max_half: float = config.CUTOUT_MAX_HALF_BEAMS,
) -> np.ndarray:
    """Half-window in beams that subtends `window_m` metres at range r0.

    arctan, not the small-angle w/(2 r dtheta), because at 0.5 m the window is
    wide enough for the difference to matter.
    """
    r0 = np.asarray(r0, dtype=float)
    half_rad = np.arctan2(window_m / 2.0, np.maximum(r0, 1e-3))
    return np.clip(half_rad / _ANG_RES_RAD, min_half, max_half)


def cutout_lateral_offsets(
    n_samples: int = config.CUTOUT_N_SAMPLES, window_m: float = config.CUTOUT_WIDTH_M
) -> np.ndarray:
    """Physical lateral offset, in metres, of each cutout sample.

    Because the window scales with range, this mapping is the same for every
    cutout regardless of distance. That is what makes it meaningful to talk
    about "the middle of the cutout" as the object and "the flanks" as
    background, which is exactly the split Part E attributes performance to.
    """
    return np.linspace(-1.0, 1.0, n_samples) * (window_m / 2.0)


def normalised_cutouts(
    ranges: np.ndarray,
    centers: np.ndarray,
    invalid: np.ndarray | None = None,
    n_samples: int = config.CUTOUT_N_SAMPLES,
    window_m: float = config.CUTOUT_WIDTH_M,
    depth_m: float = config.CUTOUT_DEPTH_M,
) -> np.ndarray:
    """(K, n_samples) scale- and offset-normalised cutouts for one frame.

    For each candidate beam c with range r0:
      1. take the angular window that subtends `window_m` metres at r0,
      2. resample it to `n_samples` points (nearest beam),
      3. subtract r0 and clip to +/- depth_m, then divide by depth_m.

    Missing readings and samples that fall off the end of the scan are set to
    +depth_m, i.e. "nothing there, far away". They are emphatically not
    averaged in: a single 29.96 inside a window would otherwise dominate every
    statistic computed from it.
    """
    ranges = np.asarray(ranges, dtype=float)
    centers = np.atleast_1d(np.asarray(centers, dtype=int))
    n_beams = ranges.shape[-1]
    if invalid is None:
        invalid = np.zeros(n_beams, dtype=bool)

    r0 = ranges[centers]
    half = cutout_half_beams(r0, window_m)
    offs = np.linspace(-1.0, 1.0, n_samples)

    idx = np.rint(centers[:, None] + offs[None, :] * half[:, None]).astype(int)
    oob = (idx < 0) | (idx >= n_beams)
    idx_c = np.clip(idx, 0, n_beams - 1)

    vals = ranges[idx_c]
    bad = oob | invalid[idx_c]
    vals = np.where(bad, r0[:, None] + depth_m, vals)
    return np.clip(vals - r0[:, None], -depth_m, depth_m) / depth_m


def raw_window_cutouts(
    ranges: np.ndarray,
    centers: np.ndarray,
    invalid: np.ndarray | None = None,
    half_beams: int = config.RAW_WINDOW_HALF_BEAMS,
    depth_m: float = config.CUTOUT_DEPTH_M,
) -> np.ndarray:
    """Control representation: fixed BEAM window, otherwise identical.

    Still centred and depth-normalised, so the only difference from
    `normalised_cutouts` is that the angular extent does not adapt to range.
    Any gap between the two is attributable to the scaling alone.
    """
    ranges = np.asarray(ranges, dtype=float)
    centers = np.atleast_1d(np.asarray(centers, dtype=int))
    n_beams = ranges.shape[-1]
    if invalid is None:
        invalid = np.zeros(n_beams, dtype=bool)

    offs = np.arange(-half_beams, half_beams + 1)
    idx = centers[:, None] + offs[None, :]
    oob = (idx < 0) | (idx >= n_beams)
    idx_c = np.clip(idx, 0, n_beams - 1)

    r0 = ranges[centers]
    vals = ranges[idx_c]
    vals = np.where(oob | invalid[idx_c], r0[:, None] + depth_m, vals)
    return np.clip(vals - r0[:, None], -depth_m, depth_m) / depth_m


# ---------------------------------------------------------------------------
# Classical geometric features (Arras et al., ICRA 2007)
# ---------------------------------------------------------------------------
ARRAS_NAMES = (
    "n_valid",
    "width_m",
    "std_from_centroid",
    "mad_from_median_r",
    "linearity_resid",
    "circularity_resid",
    "circle_radius_m",
    "boundary_length_m",
    "boundary_regularity",
    "mean_curvature",
    "centre_range_m",
    "depth_extent_m",
    "missing_frac",
    "mean_range_offset_m",
)


def _batched_circle_fit(x, y, w):
    """Kasa least-squares circle fit, batched over K windows.

    Returns (radius, mean absolute radial residual). A ridge term keeps the
    3x3 normal equations solvable when the window is a flat wall, which is the
    single most common case in a corridor.
    """
    A = np.stack([x, y, np.ones_like(x)], axis=-1)           # (K, S, 3)
    b = (x**2 + y**2)[..., None]                             # (K, S, 1)
    W = w[..., None]                                         # (K, S, 1)
    AtA = np.einsum("ksi,ksj->kij", A * W, A)
    Atb = np.einsum("ksi,ksj->kij", A * W, b)[..., 0]
    ridge = 1e-6 * np.maximum(np.trace(AtA, axis1=1, axis2=2) / 3.0, 1e-9)
    AtA = AtA + ridge[:, None, None] * np.eye(3)
    try:
        theta = np.linalg.solve(AtA, Atb[..., None])[..., 0]
    except np.linalg.LinAlgError:
        theta = (np.linalg.pinv(AtA) @ Atb[..., None])[..., 0]

    cx, cy = theta[:, 0] / 2.0, theta[:, 1] / 2.0
    rad2 = theta[:, 2] + cx**2 + cy**2
    rad = np.sqrt(np.maximum(rad2, 0.0))
    d = np.sqrt((x - cx[:, None]) ** 2 + (y - cy[:, None]) ** 2)
    resid = np.sum(np.abs(d - rad[:, None]) * w, axis=1) / np.maximum(w.sum(1), 1)
    return rad, resid


def arras_features(
    ranges: np.ndarray,
    centers: np.ndarray,
    invalid: np.ndarray | None = None,
    half_beams: int = config.RAW_WINDOW_HALF_BEAMS,
) -> np.ndarray:
    """(K, 14) geometric descriptors of the neighbourhood of each candidate beam.

    Computed in metres on the Cartesian points, so most of these are already
    range-invariant for a physically fixed object -- which is precisely the
    argument for comparing them against the normalised cutout rather than
    against raw beams.
    """
    ranges = np.asarray(ranges, dtype=float)
    centers = np.atleast_1d(np.asarray(centers, dtype=int))
    n_beams = ranges.shape[-1]
    if invalid is None:
        invalid = np.zeros(n_beams, dtype=bool)

    offs = np.arange(-half_beams, half_beams + 1)
    idx = centers[:, None] + offs[None, :]
    oob = (idx < 0) | (idx >= n_beams)
    idx_c = np.clip(idx, 0, n_beams - 1)

    w = (~(oob | invalid[idx_c])).astype(float)              # (K, S)
    nvalid = w.sum(1)
    r = np.where(w > 0, ranges[idx_c], np.nan)
    phi = np.asarray(config.BEAM_ANGLES_RAD)[idx_c]
    x, y = r * np.cos(phi), r * np.sin(phi)
    x0, y0 = np.nan_to_num(x), np.nan_to_num(y)

    r0 = ranges[centers]
    denom = np.maximum(nvalid, 1)
    cx = (x0 * w).sum(1) / denom
    cy = (y0 * w).sum(1) / denom

    # 3: spread about the centroid
    d_cent = np.sqrt((x0 - cx[:, None]) ** 2 + (y0 - cy[:, None]) ** 2)
    std_cent = np.sqrt(((d_cent**2) * w).sum(1) / denom)

    # 4: robust radial spread -- median is immune to one stray return
    med_r = np.nanmedian(r, axis=1)
    mad_r = np.nansum(np.abs(r - med_r[:, None]), axis=1) / denom

    # 2: width, first valid point to last valid point
    first = np.argmax(w > 0, axis=1)
    last = w.shape[1] - 1 - np.argmax(w[:, ::-1] > 0, axis=1)
    k = np.arange(len(centers))
    width = np.hypot(x0[k, last] - x0[k, first], y0[k, last] - y0[k, first])

    # 5: linearity -- the smaller principal eigenvalue of the point covariance
    dx, dy = (x0 - cx[:, None]) * w, (y0 - cy[:, None]) * w
    sxx = (dx * dx).sum(1) / denom
    syy = (dy * dy).sum(1) / denom
    sxy = (dx * dy).sum(1) / denom
    tr, det = sxx + syy, sxx * syy - sxy**2
    disc = np.sqrt(np.maximum(tr**2 / 4.0 - det, 0.0))
    lin = np.sqrt(np.maximum(tr / 2.0 - disc, 0.0))

    # 6, 7: circularity
    rad, circ_resid = _batched_circle_fit(x0, y0, w)

    # 8, 9: boundary length and its regularity
    seg = np.hypot(np.diff(x0, axis=1), np.diff(y0, axis=1))
    segw = w[:, 1:] * w[:, :-1]
    nseg = np.maximum(segw.sum(1), 1)
    blen = (seg * segw).sum(1)
    bmean = blen / nseg
    breg = np.sqrt((((seg - bmean[:, None]) ** 2) * segw).sum(1) / nseg)

    # 10: mean curvature from the circumscribed circle of each consecutive triple
    p0, p1, p2 = (x0[:, :-2], y0[:, :-2]), (x0[:, 1:-1], y0[:, 1:-1]), (x0[:, 2:], y0[:, 2:])
    a = np.hypot(p1[0] - p0[0], p1[1] - p0[1])
    bq = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
    c = np.hypot(p2[0] - p0[0], p2[1] - p0[1])
    area = 0.5 * np.abs(
        (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])
    )
    triw = w[:, :-2] * w[:, 1:-1] * w[:, 2:]
    curv = 4.0 * area / np.maximum(a * bq * c, 1e-9)
    mcurv = (curv * triw).sum(1) / np.maximum(triw.sum(1), 1)

    depth_extent = np.nanmax(r, axis=1) - np.nanmin(r, axis=1)
    mean_off = np.nanmean(r, axis=1) - r0

    feats = np.stack(
        [
            nvalid,
            width,
            std_cent,
            mad_r,
            lin,
            circ_resid,
            rad,
            blen,
            breg,
            mcurv,
            r0,
            depth_extent,
            1.0 - nvalid / w.shape[1],
            mean_off,
        ],
        axis=1,
    )
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# Jump-distance segmentation (feeds the geometric baseline in Part D)
# ---------------------------------------------------------------------------
def segment_scan(
    ranges: np.ndarray,
    valid: np.ndarray,
    c0: float = 0.10,
) -> list[np.ndarray]:
    """Split one scan into connected segments by adaptive jump distance.

    Threshold from Dietmayer: C0 + C1 * min(r_i, r_{i+1}) with
    C1 = sqrt(2 (1 - cos dtheta)), i.e. the distance two adjacent beams would
    naturally be apart on a surface at that range. A constant threshold either
    shatters far objects or merges near ones; this one does neither.
    """
    ranges = np.asarray(ranges, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    c1 = np.sqrt(2.0 * (1.0 - np.cos(_ANG_RES_RAD)))

    x, y = L.scan_to_cartesian(ranges)
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return []

    d = np.hypot(np.diff(x[idx]), np.diff(y[idx]))
    thr = c0 + c1 * np.minimum(ranges[idx][:-1], ranges[idx][1:])
    # a gap in beam index is also a break: we must not bridge a dropout region
    contiguous = np.diff(idx) <= 2
    breaks = np.flatnonzero((d > thr) | (~contiguous)) + 1
    return [s for s in np.split(idx, breaks) if s.size]


def segment_descriptors(ranges: np.ndarray, segments: list[np.ndarray]) -> np.ndarray:
    """(n_segments, 5): n_points, width, depth, centre range, centre bearing."""
    if not segments:
        return np.zeros((0, 5))
    x, y = L.scan_to_cartesian(ranges)
    rows = []
    for s in segments:
        w = np.hypot(x[s[-1]] - x[s[0]], y[s[-1]] - y[s[0]])
        # sagitta: how far the middle of the segment bulges off its own chord
        if s.size >= 3:
            vx, vy = x[s[-1]] - x[s[0]], y[s[-1]] - y[s[0]]
            n = np.hypot(vx, vy) + 1e-9
            depth = np.max(
                np.abs((x[s] - x[s[0]]) * (-vy / n) + (y[s] - y[s[0]]) * (vx / n))
            )
        else:
            depth = 0.0
        cx, cy = x[s].mean(), y[s].mean()
        rows.append([s.size, w, depth, np.hypot(cx, cy), np.arctan2(cy, cx)])
    return np.asarray(rows)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
REPRESENTATIONS = ("cutout", "raw_window", "arras")


def extract(
    representation: str,
    ranges: np.ndarray,
    centers: np.ndarray,
    invalid: np.ndarray | None = None,
) -> np.ndarray:
    if representation == "cutout":
        return normalised_cutouts(ranges, centers, invalid)
    if representation == "raw_window":
        return raw_window_cutouts(ranges, centers, invalid)
    if representation == "arras":
        return arras_features(ranges, centers, invalid)
    raise ValueError(f"unknown representation {representation!r}")


def feature_names(representation: str) -> list[str]:
    if representation == "cutout":
        return [f"off_{o:+.3f}m" for o in cutout_lateral_offsets()]
    if representation == "raw_window":
        h = config.RAW_WINDOW_HALF_BEAMS
        return [f"beam_{o:+d}" for o in range(-h, h + 1)]
    if representation == "arras":
        return list(ARRAS_NAMES)
    raise ValueError(f"unknown representation {representation!r}")
