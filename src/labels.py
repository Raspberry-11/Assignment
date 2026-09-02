"""Label parsing, the three-state frame index, and polar/beam geometry.

Labels cover about 5% of frames, so plain Python is the honest tool here: the
whole person-label corpus is a few tens of thousands of lines. Spark would add
a scheduler to a job that finishes in under a second.

Two things in this module are load-bearing and easy to get silently wrong:

1. `seq,<JSON list>` must be split on the FIRST comma only. The payload is a
   JSON list of pairs and is full of commas; `line.split(",")` mangles it.
2. An absent sequence number is not a negative. `frame_states()` keeps
   annotated-positive, annotated-empty and never-annotated apart, and every
   count downstream is derived from it rather than from a boolean.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


# ---------------------------------------------------------------------------
# Beam geometry
# ---------------------------------------------------------------------------
# config.BEAM_ANGLES_RAD is an INCREASING ramp: beam 0 sits at -112.5 deg (the
# robot's right) and beam 449 at +112.5 deg (its left). The brief claims the
# opposite. Section B2 of the notebook scores both conventions against the
# annotations rather than trusting either, and this is the one the data picks.
_PHI_MIN = -np.radians(config.FOV_DEG / 2.0)
_PHI_MAX = np.radians(config.FOV_DEG / 2.0)
_STEP = (_PHI_MAX - _PHI_MIN) / (config.N_BEAMS - 1)


def beam_phi(index):
    """Bearing (rad) of a beam index. Inverse of `phi_to_beam`."""
    return np.asarray(config.BEAM_ANGLES_RAD)[np.asarray(index)]


def phi_to_beam(phi, clip: bool = False):
    """Nearest beam index for a bearing in radians.

    Closed form rather than a search: the beam grid is uniform, so the index is
    just how many half-degree steps we are in from the left edge. Bearings
    outside the 225 deg field of view return -1 unless `clip` is set, because
    an object annotated outside the FOV is a data-quality event worth counting,
    not something to quietly snap to beam 0.
    """
    phi = np.asarray(phi, dtype=float)
    idx = np.rint((phi - _PHI_MIN) / _STEP).astype(int)
    if clip:
        return np.clip(idx, 0, config.N_BEAMS - 1)
    out = np.where((idx >= 0) & (idx < config.N_BEAMS), idx, -1)
    return out if out.ndim else int(out)


def polar_to_cartesian(r, phi):
    """(r, phi) -> (x, y) in the sensor frame: x forward, y to the robot's left.

    ROS REP-103 convention. Plotting helpers flip the horizontal axis so that
    the robot's left appears on the left of the page; see src.viz.
    """
    r = np.asarray(r, dtype=float)
    phi = np.asarray(phi, dtype=float)
    return r * np.cos(phi), r * np.sin(phi)


def scan_to_cartesian(ranges):
    """Whole 450-vector (or an (n, 450) stack) to Cartesian points."""
    ranges = np.asarray(ranges, dtype=float)
    return polar_to_cartesian(ranges, config.BEAM_ANGLES_RAD)


def expected_beams_on_object(width_m: float, r_m):
    """w / (r * dtheta): how many beams land on an object of width w at range r.

    The single most important number in this assignment. It is what forces
    distance-normalised features and range-stratified evaluation.
    """
    r_m = np.asarray(r_m, dtype=float)
    return width_m / (r_m * np.radians(config.ANGULAR_RES_DEG))


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def parse_label_file(path: Path) -> list[tuple[int, list[tuple[float, float]]]]:
    """Parse one .wc/.wa/.wp file into [(seq, [(r, phi), ...]), ...].

    An empty payload list is preserved as an empty list, and is meaningful: it
    is a human asserting absence. It is not the same as the sequence number
    being missing from the file.
    """
    records = []
    with open(path, "r") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            # partition, not split: the JSON payload contains commas.
            seq_str, sep, payload = line.partition(",")
            if not sep:
                raise ValueError(f"{path}:{lineno}: no comma in label line {line[:60]!r}")
            try:
                seq = int(seq_str)
                objects = json.loads(payload) if payload.strip() else []
            except (ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path}:{lineno}: unparseable {line[:60]!r}") from exc
            records.append((seq, [(float(r), float(phi)) for r, phi in objects]))
    return records


def load_labels(
    cls: str = config.TARGET_CLASS,
    splits: tuple[str, ...] = config.SPLITS,
    data_root: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read every label file of one class.

    Returns (annotated, objects):
      annotated -- one row per ANNOTATED frame: split, file_id, seq, n_objects.
                   n_objects == 0 marks an explicit-empty frame.
      objects   -- one row per annotated object: split, file_id, seq, obj_idx,
                   r, phi, plus derived beam index and Cartesian position.

    Frames absent from both tables were never looked at by a human. That
    distinction is materialised by `frame_states()`.
    """
    data_root = Path(data_root or config.DATA_ROOT)
    ann_rows, obj_rows = [], []

    for split in splits:
        for path in sorted((data_root / split).glob(f"*.{cls}")):
            file_id = path.name[: -(len(cls) + 1)]
            if file_id.endswith(".bag"):
                file_id = file_id[:-4]
            for seq, objects in parse_label_file(path):
                ann_rows.append((split, file_id, seq, len(objects)))
                for j, (r, phi) in enumerate(objects):
                    obj_rows.append((split, file_id, seq, j, r, phi))

    annotated = pd.DataFrame(ann_rows, columns=["split", "file_id", "seq", "n_objects"])
    objects = pd.DataFrame(
        obj_rows, columns=["split", "file_id", "seq", "obj_idx", "r", "phi"]
    )
    if len(objects):
        objects["beam"] = phi_to_beam(objects["phi"].to_numpy())
        x, y = polar_to_cartesian(objects["r"].to_numpy(), objects["phi"].to_numpy())
        objects["x"], objects["y"] = x, y
    else:  # keep the schema stable even for an empty class/split selection
        for col in ("beam", "x", "y"):
            objects[col] = pd.Series(dtype=float)
    return annotated, objects


def load_all_classes(
    splits: tuple[str, ...] = config.SPLITS, data_root: Path | None = None
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """All three classes, for the comparative label EDA in B3."""
    return {cls: load_labels(cls, splits, data_root) for cls in config.LABEL_EXTENSIONS}


# ---------------------------------------------------------------------------
# The three-state frame index
# ---------------------------------------------------------------------------
def frame_states(scan_frames: pd.DataFrame, annotated: pd.DataFrame) -> pd.DataFrame:
    """Label every (split, file_id, seq) in the scan corpus with one of three states.

      positive -- annotated, at least one object of this class present
      empty    -- annotated, human asserted no object of this class
      absent   -- never annotated; carries NO information about the class

    `scan_frames` is the universe of frames that actually exist in the scan
    files, taken from the Parquet cache. Deriving the universe from the label
    files instead would make 'absent' undiscoverable.
    """
    keys = ["split", "file_id", "seq"]
    out = scan_frames[keys].merge(annotated, on=keys, how="left")
    out["state"] = np.where(
        out["n_objects"].isna(),
        "absent",
        np.where(out["n_objects"].to_numpy() > 0, "positive", "empty"),
    )
    return out


def state_summary(states: pd.DataFrame) -> pd.DataFrame:
    """Frame counts per split per state, with the annotated fraction."""
    tab = (
        states.pivot_table(index="split", columns="state", aggfunc="size", fill_value=0)
        .reindex(columns=["positive", "empty", "absent"], fill_value=0)
        .reindex(list(config.SPLITS))
    )
    tab["frames"] = tab.sum(axis=1)
    tab["annotated"] = tab["positive"] + tab["empty"]
    tab["annotated_frac"] = tab["annotated"] / tab["frames"]
    return tab


def annotation_gaps(annotated: pd.DataFrame) -> pd.DataFrame:
    """Gaps between consecutive annotated sequence numbers, per file.

    The documented regime -- every 5th frame within every 4th batch of 100 --
    predicts exactly two gap values, 5 and 305. This is how that prediction is
    tested rather than repeated.
    """
    rows = []
    for (split, file_id), grp in annotated.groupby(["split", "file_id"]):
        seqs = np.sort(grp["seq"].unique())
        if len(seqs) < 2:
            continue
        gaps = np.diff(seqs)
        vals, counts = np.unique(gaps, return_counts=True)
        rows.append(
            {
                "split": split,
                "file_id": file_id,
                "n_annotated": len(seqs),
                "distinct_gaps": len(vals),
                "gap_values": ",".join(map(str, vals[:6])),
                "modal_gap": int(vals[np.argmax(counts)]),
                "matches_5_305": set(vals.tolist()) <= {5, 305},
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-beam labels
# ---------------------------------------------------------------------------
def beam_labels(
    ranges: np.ndarray,
    obj_r: np.ndarray,
    obj_phi: np.ndarray,
    radius_m: float = config.PERSON_RADIUS_M,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """Boolean mask over 450 beams: does this beam land on an annotated person?

    A beam is positive when its measured Cartesian endpoint is within
    `radius_m` of an annotated object centre. Using the measured endpoint --
    rather than declaring an angular window positive regardless of range -- is
    what stops beams that merely pass *near* a person in bearing but hit the
    wall five metres behind them from being labelled positive.

    Beams with no return are never positive: there is nothing there to hit.
    """
    ranges = np.asarray(ranges, dtype=float)
    if valid is None:
        valid = np.ones_like(ranges, dtype=bool)
    bx, by = scan_to_cartesian(ranges)
    mask = np.zeros(ranges.shape[-1], dtype=bool)
    if len(obj_r) == 0:
        return mask
    ox, oy = polar_to_cartesian(np.asarray(obj_r), np.asarray(obj_phi))
    d2 = (bx[:, None] - ox[None, :]) ** 2 + (by[:, None] - oy[None, :]) ** 2
    mask = (d2 <= radius_m**2).any(axis=1)
    return mask & valid


def radius_sensitivity(
    R: np.ndarray,
    valid: np.ndarray,
    objects_per_frame: list[np.ndarray],
    radii=(0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60),
) -> pd.DataFrame:
    """How the positive-beam count responds to the association radius.

    The radius is a modelling choice, not a fact about the data, so it needs a
    sensitivity curve rather than a citation. A good radius sits on the plateau
    where nearly every object has acquired its beams but the count has not yet
    started growing linearly by swallowing background.
    """
    rows = []
    for rad in radii:
        n_pos, n_obj_hit, n_obj = 0, 0, 0
        for i, objs in enumerate(objects_per_frame):
            if len(objs) == 0:
                continue
            m = beam_labels(R[i], objs[:, 0], objs[:, 1], rad, valid[i])
            n_pos += int(m.sum())
            bx, by = scan_to_cartesian(R[i])
            ox, oy = polar_to_cartesian(objs[:, 0], objs[:, 1])
            d2 = (bx[valid[i]][:, None] - ox[None, :]) ** 2 + (
                by[valid[i]][:, None] - oy[None, :]
            ) ** 2
            n_obj_hit += int((d2.min(axis=0) <= rad**2).sum()) if d2.size else 0
            n_obj += len(objs)
        rows.append(
            {
                "radius_m": rad,
                "positive_beams": n_pos,
                "objects_with_a_beam": n_obj_hit,
                "objects": n_obj,
                "object_coverage": n_obj_hit / max(n_obj, 1),
                "beams_per_object": n_pos / max(n_obj_hit, 1),
            }
        )
    return pd.DataFrame(rows)
