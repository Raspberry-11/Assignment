"""Assembling model-ready matrices from the Parquet cache and the label files.

The split of labour defended in the notebook: Spark owns everything that
touches all 464k frames, and hands over exactly the ~23k annotated frames as a
dense numpy array (23k x 450 float64 is under 100 MB). From there it is
sklearn. Running a random forest through Spark ML at this size would be
engineering theatre.

Two rules are enforced structurally rather than by convention:

  * only ANNOTATED frames ever become training or evaluation examples, so a
    never-annotated frame can never be mistaken for a negative;
  * negative subsampling happens in TRAINING ONLY. Evaluation scores every
    candidate beam of every annotated evaluation frame, because a PR curve
    computed on subsampled negatives reports a precision that does not exist.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.sql import functions as SF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src import features as FE  # noqa: E402
from src import io as dio  # noqa: E402
from src import labels as L  # noqa: E402

KEYS = ["split", "file_id", "seq"]


@dataclass
class FrameSet:
    """One split's annotated frames, materialised.

    keys    -- (n,) rows of split/file_id/seq, sorted, so the order is stable
               across runs and machines.
    R       -- (n, 450) raw ranges, sentinels included and untouched.
    missing -- (n, 450) sentinel mask.
    valid   -- (n, 450) candidate mask (not missing, inside the range envelope).
    objects -- length-n list of (k, 2) arrays of annotated (r, phi).
    """

    keys: pd.DataFrame
    R: np.ndarray
    missing: np.ndarray
    valid: np.ndarray
    objects: list

    def __len__(self) -> int:
        return len(self.keys)

    @property
    def n_objects(self) -> int:
        return int(sum(len(o) for o in self.objects))

    def subset(self, idx) -> "FrameSet":
        idx = np.asarray(idx)
        return FrameSet(
            self.keys.iloc[idx].reset_index(drop=True),
            self.R[idx],
            self.missing[idx],
            self.valid[idx],
            [self.objects[i] for i in idx],
        )


# ---------------------------------------------------------------------------
# Pulling annotated frames out of the cache
# ---------------------------------------------------------------------------
def annotated_keys(
    annotated: pd.DataFrame, split: str, stride: int = 1
) -> pd.DataFrame:
    """Sorted (split, file_id, seq) for one split's annotated frames.

    `stride` keeps every Nth annotated frame per recording. Consecutive
    annotated frames within a batch are 5 apart, i.e. 0.4 s, and are close to
    duplicates of each other; thinning them costs little and halves the
    training set. It is applied per recording so it never removes a whole file.
    """
    sel = annotated.loc[annotated["split"] == split, KEYS].sort_values(KEYS)
    if stride > 1:
        rank = sel.groupby("file_id").cumcount()
        sel = sel.loc[rank % stride == 0]
    return sel.reset_index(drop=True)


def load_frameset(
    spark,
    keys: pd.DataFrame,
    objects: pd.DataFrame,
    sentinels,
    cache_dir: Path | None = None,
) -> FrameSet:
    """Join the requested keys against the Parquet cache and densify.

    The join is broadcast: the key table is tens of thousands of rows against a
    scan table of hundreds of thousands, and shuffling the big side would be
    the expensive way to do a lookup.
    """
    scans = dio.read_cache(spark, "scans", cache_dir)
    kdf = spark.createDataFrame(keys[KEYS])
    pdf = (
        scans.join(SF.broadcast(kdf), on=KEYS, how="inner")
        .select(*KEYS, "ranges")
        .toPandas()
        .sort_values(KEYS)          # Spark returns partitions in arbitrary order
        .reset_index(drop=True)
    )
    if len(pdf) != len(keys):
        missing_n = len(keys) - len(pdf)
        raise ValueError(
            f"{missing_n} annotated frames have no scan row in the cache. "
            "Check the orphan-recording report from Part A before proceeding."
        )

    R = np.asarray(pdf["ranges"].tolist(), dtype=np.float64)
    miss = FE.missing_mask(R, sentinels)
    valid = FE.candidate_mask(R, miss)

    by_frame = {
        k: g[["r", "phi"]].to_numpy()
        for k, g in objects.groupby(KEYS, sort=False)
    }
    empty = np.zeros((0, 2))
    objs = [
        by_frame.get(tuple(row), empty)
        for row in pdf[KEYS].itertuples(index=False, name=None)
    ]
    return FrameSet(pdf[KEYS], R, miss, valid, objs)


def save_frameset(fs: FrameSet, path: Path) -> None:
    """Cache a FrameSet to .npz so a restart-and-run-all does not re-enter Spark."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = np.concatenate(fs.objects) if fs.objects else np.zeros((0, 2))
    counts = np.array([len(o) for o in fs.objects])
    np.savez_compressed(
        path,
        R=fs.R.astype(np.float32),
        missing=fs.missing,
        valid=fs.valid,
        obj_flat=flat,
        obj_counts=counts,
        split=fs.keys["split"].to_numpy(),
        file_id=fs.keys["file_id"].to_numpy(),
        seq=fs.keys["seq"].to_numpy(),
    )


def load_frameset_npz(path: Path) -> FrameSet:
    z = np.load(path, allow_pickle=True)
    keys = pd.DataFrame(
        {"split": z["split"], "file_id": z["file_id"], "seq": z["seq"]}
    )
    objs = np.split(z["obj_flat"], np.cumsum(z["obj_counts"])[:-1])
    return FrameSet(keys, z["R"].astype(np.float64), z["missing"], z["valid"], list(objs))


# ---------------------------------------------------------------------------
# Per-beam examples
# ---------------------------------------------------------------------------
def build_beam_dataset(
    fs: FrameSet,
    representation: str = "cutout",
    radius_m: float = config.PERSON_RADIUS_M,
    negative_ratio: int | None = config.NEGATIVE_RATIO,
    seed: int = config.SEED,
) -> dict:
    """Per-beam examples for one FrameSet.

    A positive is a candidate beam whose measured endpoint lies within
    `radius_m` of an annotated person centre. A negative is any other candidate
    beam in an annotated frame -- including beams on wheelchairs and walking
    aids, which are the hard negatives this task actually has to survive.

    `negative_ratio=None` keeps every candidate beam. Use that for evaluation.
    """
    rng = np.random.default_rng(seed)
    Xs, ys, frame_ids, beam_ids = [], [], [], []

    for i in range(len(fs)):
        valid = fs.valid[i]
        objs = fs.objects[i]
        pos_mask = L.beam_labels(
            fs.R[i], objs[:, 0], objs[:, 1], radius_m, valid
        ) if len(objs) else np.zeros(config.N_BEAMS, dtype=bool)

        centers = np.flatnonzero(valid)
        if centers.size == 0:
            continue
        y = pos_mask[centers]

        if negative_ratio is not None:
            pos_idx = centers[y]
            neg_pool = centers[~y]
            # A frame with no person still contributes background: explicit-empty
            # frames are evidence, and dropping them would bias the negatives
            # towards "scenes that happen to contain a person somewhere else".
            n_keep = min(neg_pool.size, max(negative_ratio * pos_idx.size, negative_ratio))
            neg_idx = rng.choice(neg_pool, size=n_keep, replace=False) if n_keep else neg_pool[:0]
            centers = np.sort(np.concatenate([pos_idx, neg_idx]))
            y = pos_mask[centers]

        Xs.append(FE.extract(representation, fs.R[i], centers, fs.missing[i]))
        ys.append(y)
        frame_ids.append(np.full(centers.size, i))
        beam_ids.append(centers)

    X = np.concatenate(Xs).astype(np.float32)
    y = np.concatenate(ys)
    frame = np.concatenate(frame_ids)
    beam = np.concatenate(beam_ids)
    meta = pd.DataFrame(
        {
            "frame": frame,
            "beam": beam,
            "r0": fs.R[frame, beam],
            "file_id": fs.keys["file_id"].to_numpy()[frame],
            "seq": fs.keys["seq"].to_numpy()[frame],
        }
    )
    return {
        "X": X,
        "y": y,
        "meta": meta,
        "groups": meta["file_id"].to_numpy(),
        "feature_names": FE.feature_names(representation),
        "representation": representation,
    }


def class_balance(ds: dict) -> pd.Series:
    """Positives, negatives and the resulting ratio. Print this, always."""
    y = ds["y"]
    return pd.Series(
        {
            "examples": y.size,
            "positives": int(y.sum()),
            "negatives": int((~y).sum()),
            "positive_rate": float(y.mean()),
            "negatives_per_positive": float((~y).sum() / max(y.sum(), 1)),
        }
    )
