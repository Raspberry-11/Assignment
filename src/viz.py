"""Plot helpers.

Everything that draws lives here so that a notebook cell is one call and one
sentence of interpretation, which is the format the brief asks for.

Plotting convention throughout: the robot sits at the origin facing up the
page, the horizontal axis is lateral offset, and the axis is inverted so that
the robot's left is on the left of the page. Getting this backwards makes every
geometric claim in the notebook read as its own mirror image.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src import features as FE  # noqa: E402
from src import labels as L  # noqa: E402


def savefig(fig, name: str, directory: Path | None = None) -> Path:
    directory = Path(directory or config.FIGURE_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    return path


def plot_frame(
    fs, i: int, ax=None, detections=None, score_threshold: float = 0.5,
    radius_m: float = config.PERSON_RADIUS_M, max_r: float = 12.0, title: str | None = None,
):
    """One annotated scan in Cartesian space, with labels and optionally predictions.

    This is the plot the whole assignment starts from. Missing returns are drawn
    in a different colour rather than dropped, because where the sensor sees
    nothing is itself information about the scene.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    R, miss, valid = fs.R[i], fs.missing[i], fs.valid[i]
    x, y = L.scan_to_cartesian(R)

    ax.scatter(y[valid], x[valid], s=4, c="0.35", label="return")
    if miss.any():
        # sentinels are drawn at the plot edge, not at 29.96 m, so they stay
        # visible without rescaling the whole figure around a magic number
        xm, ym = L.polar_to_cartesian(max_r * 0.98, config.BEAM_ANGLES_RAD[miss])
        ax.scatter(ym, xm, s=3, c="tab:orange", alpha=0.5, label="no return (sentinel)")

    objs = fs.objects[i]
    for j, (r, phi) in enumerate(objs):
        ox, oy = L.polar_to_cartesian(r, phi)
        ax.add_patch(plt.Circle((oy, ox), radius_m, fill=False, color="tab:green", lw=1.6,
                                label="annotated person" if j == 0 else None))

    if detections is not None:
        d = detections[(detections["frame"] == i) & (detections["score"] >= score_threshold)]
        ax.scatter(d["y"], d["x"], s=70, marker="x", c="tab:red", lw=1.8, label="detection")

    ax.scatter([0], [0], marker="^", s=120, c="tab:blue", label="scanner")
    ax.set_aspect("equal")
    ax.set_xlim(max_r, -max_r)          # inverted: robot's left on the left
    ax.set_ylim(-1.0, max_r)
    ax.set_xlabel("lateral offset y [m]  (robot's left <- -> right)")
    ax.set_ylabel("forward x [m]")
    key = fs.keys.iloc[i]
    ax.set_title(title or f"{key.split}/{key.file_id}  seq={key.seq}", fontsize=9)
    ax.grid(alpha=0.25)
    return ax


def plot_range_profile(fs, i: int, ax=None):
    """The same frame as a 1-D signal, which is what the model actually consumes."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 2.6))
    R, miss = fs.R[i], fs.missing[i]
    b = np.arange(config.N_BEAMS)
    ax.plot(b[~miss], R[~miss], lw=0.8, c="0.3")
    ax.scatter(b[miss], np.full(miss.sum(), R.max()), s=4, c="tab:orange")
    for r, phi in fs.objects[i]:
        beam = L.phi_to_beam(phi)
        if beam >= 0:
            ax.axvspan(beam - 8, beam + 8, color="tab:green", alpha=0.25)
    ax.set_xlabel("beam index (0 = leftmost)")
    ax.set_ylabel("range [m]")
    ax.grid(alpha=0.25)
    return ax


def plot_cutout_bank(X, y, feature_names=None, ax=None, n_show: int = 40, seed=config.SEED):
    """Positive cutouts against negative cutouts, on the physical offset axis."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    off = FE.cutout_lateral_offsets(X.shape[1])
    rng = np.random.default_rng(seed)
    for cls, colour, lbl in ((True, "tab:green", "person"), (False, "0.6", "background")):
        idx = np.flatnonzero(y == cls)
        if idx.size == 0:
            continue
        for k in rng.choice(idx, size=min(n_show, idx.size), replace=False):
            ax.plot(off, X[k], c=colour, alpha=0.12, lw=0.8)
        ax.plot(off, X[idx].mean(0), c=colour, lw=2.5, label=f"{lbl} (mean, n={idx.size})")
    ax.axhline(0, c="k", lw=0.6)
    ax.set_xlabel("lateral offset from centre beam [m]")
    ax.set_ylabel("normalised range offset")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    return ax


def plot_pr(results: dict, ax=None, title: str = "Precision-recall, person detection"):
    """Overlay PR curves. `results` maps a label to an evaluate.run_evaluation dict."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    for name, res in results.items():
        ax.plot(res["recall"], res["precision"], lw=1.8, label=f"{name}  (AP={res['ap']:.3f})")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)
    return ax


def plot_trajectory(odom_df, ax=None, max_files: int = 6):
    """Integrated odometry differences. Never the raw values: the origin is arbitrary."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    for fid, g in list(odom_df.groupby("file_id"))[:max_files]:
        g = g.sort_values("time_s")
        dx = np.diff(g["tx"].to_numpy(), prepend=g["tx"].iloc[0])
        dy = np.diff(g["ty"].to_numpy(), prepend=g["ty"].iloc[0])
        ax.plot(np.cumsum(dx), np.cumsum(dy), lw=1.0, label=fid[:22])
    ax.set_aspect("equal")
    ax.set_xlabel("displacement x [m]")
    ax.set_ylabel("displacement y [m]")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    return ax


def plot_failure_grid(fs, frames, detections, score_threshold, ncols: int = 3, max_r: float = 10.0):
    """A grid of the worst frames, ground truth and predictions overlaid."""
    frames = list(frames)
    nrows = int(np.ceil(len(frames) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.6 * nrows))
    for ax, i in zip(np.atleast_1d(axes).ravel(), frames):
        plot_frame(fs, int(i), ax=ax, detections=detections,
                   score_threshold=score_threshold, max_r=max_r)
        ax.legend().set_visible(False)
    for ax in np.atleast_1d(axes).ravel()[len(frames):]:
        ax.axis("off")
    fig.tight_layout()
    return fig
