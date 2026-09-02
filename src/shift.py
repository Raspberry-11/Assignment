"""Part E: attributing the train->test drop.

Hypothesis, stated before any of this was run:

    The train->test performance drop is driven mainly by BACKGROUND CORRIDOR
    GEOMETRY, not by a change in how people appear to the sensor.

The test is by attribution, in three independent legs, because a single
ablation cannot distinguish "this augmentation helps" from "this augmentation
helps for the reason I claimed":

  1. Quantify the shift. Train a discriminator to tell a train frame from a
     test frame, report its AUC under recording-wise cross-validation, and read
     off WHICH parts of the scan it uses.
  2. Apply four augmentations in isolation, two aimed at the background and two
     aimed at the object, and measure which recover test AP.
  3. Blank the flanks of the cutout at inference and see whether test AP moves.
     If the model is leaning on background, removing background must change
     something; if it is not, this is a no-op and the hypothesis is dead.

If the background-targeting interventions do not win, that is the answer and it
gets written down as the answer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src import dataset as DS  # noqa: E402
from src import features as FE  # noqa: E402
from src import labels as L  # noqa: E402


# ---------------------------------------------------------------------------
# Leg 1: quantifying the shift
# ---------------------------------------------------------------------------
def frame_descriptors(fs, n_bins: int = 45) -> tuple[np.ndarray, list[str]]:
    """A compact, order-preserving summary of each frame.

    The scan is reduced to `n_bins` angular sectors (median range per sector,
    which ignores sentinels rather than averaging them in), plus a handful of
    whole-frame statistics. Keeping the angular axis is the point: a
    discriminator built on this can say WHERE the two distributions differ, and
    that is the evidence the hypothesis turns on.
    """
    R = np.where(fs.missing, np.nan, fs.R)
    n, nb = R.shape
    per = nb // n_bins
    sect = np.nanmedian(R[:, : per * n_bins].reshape(n, n_bins, per), axis=2)
    sect = np.nan_to_num(sect, nan=config.CANDIDATE_MAX_RANGE_M)

    extra = np.stack(
        [
            fs.missing.mean(axis=1),
            np.nan_to_num(np.nanmedian(R, axis=1), nan=0.0),
            np.nan_to_num(np.nanpercentile(R, 10, axis=1), nan=0.0),
            np.nan_to_num(np.nanpercentile(R, 90, axis=1), nan=0.0),
            np.nan_to_num(np.nanstd(R, axis=1), nan=0.0),
        ],
        axis=1,
    )
    names = [f"sector_{k:02d}" for k in range(n_bins)] + [
        "missing_frac", "median_r", "p10_r", "p90_r", "std_r"
    ]
    return np.concatenate([sect, extra], axis=1), names


def train_test_discriminator(
    fs_a, fs_b, label_a: str = "train", label_b: str = "test",
    n_splits: int = 4, seed: int = config.SEED,
) -> dict:
    """Can a classifier tell which split a frame came from?

    Cross-validated by RECORDING, not by frame. A frame-wise split would let
    the model memorise a corridor from one of its own frames and report an AUC
    near 1.0 that means nothing. Grouped by recording, the AUC answers the
    question we actually care about: is an unseen test corridor recognisably
    different from an unseen train corridor?

    AUC ~ 0.5 would mean the splits are exchangeable and there is no shift to
    explain. Anything close to 1.0 means the geometry is trivially separable.
    """
    Xa, names = frame_descriptors(fs_a)
    Xb, _ = frame_descriptors(fs_b)
    X = np.concatenate([Xa, Xb])
    y = np.concatenate([np.zeros(len(Xa)), np.ones(len(Xb))])
    groups = np.concatenate([
        fs_a.keys["file_id"].to_numpy(), fs_b.keys["file_id"].to_numpy()
    ])

    n_splits = min(n_splits, len(np.unique(groups)))
    oof = np.zeros(len(y))
    importances = np.zeros(X.shape[1])
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        clf = RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5, n_jobs=config.RF_N_JOBS,
            class_weight="balanced", random_state=seed,
        ).fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
        importances += clf.feature_importances_ / n_splits

    return {
        "auc": float(roc_auc_score(y, oof)),
        "n_a": len(Xa), "n_b": len(Xb),
        "label_a": label_a, "label_b": label_b,
        "importances": pd.Series(importances, index=names).sort_values(ascending=False),
        "oof": oof, "y": y, "groups": groups,
    }


def object_appearance_comparison(
    fs_a, fs_b, label_a: str = "train", label_b: str = "test",
    radius_m: float = config.PERSON_RADIUS_M,
) -> pd.DataFrame:
    """How people themselves look in each split, independent of their surroundings.

    The control for leg 1. If the shift were about people rather than
    corridors, these distributions -- range, angular width, beams on target,
    depth -- would differ too. They are computed from the normalised cutouts of
    positive beams, i.e. exactly what the classifier sees.
    """
    rows = []
    for name, fs in ((label_a, fs_a), (label_b, fs_b)):
        for i in range(len(fs)):
            objs = fs.objects[i]
            if not len(objs):
                continue
            bx, by = L.scan_to_cartesian(fs.R[i])
            for r, phi in objs:
                ox, oy = L.polar_to_cartesian(r, phi)
                on = (np.hypot(bx - ox, by - oy) <= radius_m) & fs.valid[i]
                if not on.any():
                    rows.append({"split": name, "r": r, "beams_on": 0,
                                 "angular_width_deg": np.nan, "depth_m": np.nan})
                    continue
                idx = np.flatnonzero(on)
                rows.append({
                    "split": name,
                    "r": float(r),
                    "beams_on": int(on.sum()),
                    "angular_width_deg": float((idx[-1] - idx[0] + 1) * config.ANGULAR_RES_DEG),
                    "depth_m": float(fs.R[i][idx].max() - fs.R[i][idx].min()),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Leg 2: augmentations, in isolation
# ---------------------------------------------------------------------------
def object_beam_mask(fs, radius_m: float = config.PERSON_RADIUS_M) -> np.ndarray:
    """(n, 450) mask of beams that land on an annotated person.

    Splitting the scan into object beams and background beams is what makes an
    augmentation "background-targeting" or "object-targeting" rather than just
    "noise". Without this mask the whole experiment is uninterpretable.
    """
    M = np.zeros(fs.R.shape, dtype=bool)
    for i in range(len(fs)):
        objs = fs.objects[i]
        if len(objs):
            M[i] = L.beam_labels(fs.R[i], objs[:, 0], objs[:, 1], radius_m, fs.valid[i])
    return M


def _smooth_field(n_rows, n_beams, rng, corr_beams=40.0):
    """Low-frequency random field along the angular axis.

    White noise on a range vector is sensor noise. A corridor that is a bit
    wider, or angled a bit differently, is a SMOOTH perturbation. Getting this
    distinction right is what separates the background augmentation from the
    sensor-noise one.
    """
    z = rng.normal(size=(n_rows, n_beams))
    k = int(max(corr_beams, 1))
    kern = np.exp(-0.5 * (np.arange(-3 * k, 3 * k + 1) / k) ** 2)
    kern /= np.sqrt((kern**2).sum())
    return np.apply_along_axis(lambda v: np.convolve(v, kern, mode="same"), 1, z)


def augment(fs, kind: str, seed: int = config.SEED, obj_mask: np.ndarray | None = None):
    """Return a new FrameSet with one augmentation applied. Never mutates `fs`.

    kind:
      mirror        -- exact left-right reflection. The FOV is symmetric about
                       straight ahead, so beam i maps to 449-i and phi to -phi
                       with no interpolation and no approximation. Geometry
                       augmentation; targets neither object nor background
                       specifically, which makes it the natural control.
      background    -- smooth low-frequency perturbation of BACKGROUND beams
                       only. Simulates a corridor of a different width and
                       shape while leaving every person untouched.
      occlusion     -- contiguous dropout on OBJECT beams. Simulates a person
                       partly hidden behind furniture or another person.
      sensor_noise  -- iid range noise on all beams at the sensor's own error
                       scale (~30 mm). Changes how any surface appears without
                       changing where anything is.
    """
    rng = np.random.default_rng(seed)
    R = fs.R.copy()
    missing = fs.missing.copy()
    objects = [o.copy() for o in fs.objects]
    keys = fs.keys.copy()

    if kind == "mirror":
        R = R[:, ::-1].copy()
        missing = missing[:, ::-1].copy()
        objects = [np.stack([o[:, 0], -o[:, 1]], axis=1) if len(o) else o for o in objects]

    elif kind == "background":
        if obj_mask is None:
            obj_mask = object_beam_mask(fs)
        field = _smooth_field(len(fs), config.N_BEAMS, rng) * config.AUG_BG_PERTURB_SD_M
        where = (~obj_mask) & (~missing)
        R = np.where(where, np.maximum(R + field, config.CANDIDATE_MIN_RANGE_M), R)

    elif kind == "occlusion":
        if obj_mask is None:
            obj_mask = object_beam_mask(fs)
        # contiguous runs, not iid beams: a chair leg occludes an arc
        starts = rng.random(fs.R.shape) < (config.AUG_DROPOUT_P / 5.0)
        run = np.zeros_like(starts)
        for s in range(5):
            run[:, s:] |= starts[:, : config.N_BEAMS - s]
        hit = run & obj_mask
        R = np.where(hit, config.SENTINEL_NOMINAL, R)
        missing = missing | hit

    elif kind == "sensor_noise":
        noise = rng.normal(0.0, config.AUG_RANGE_NOISE_SD_M, fs.R.shape)
        R = np.where(~missing, np.maximum(R + noise, config.CANDIDATE_MIN_RANGE_M), R)

    else:
        raise ValueError(f"unknown augmentation {kind!r}")

    valid = FE.candidate_mask(R, missing)
    return DS.FrameSet(keys, R, missing, valid, objects)


AUGMENTATION_TARGETS = {
    "mirror": "geometry (control)",
    "background": "background",
    "sensor_noise": "object/sensor",
    "occlusion": "object",
}


def augmented_training_set(
    fs, kinds, representation: str = "cutout", seed: int = config.SEED, **kw
) -> dict:
    """Original training beams plus one augmented copy per requested kind.

    Concatenating rather than replacing is deliberate: an augmentation that
    replaced the originals would confound "this transformation helps" with
    "the original data was in the way".
    """
    parts = [DS.build_beam_dataset(fs, representation, seed=seed, **kw)]
    obj_mask = object_beam_mask(fs) if set(kinds) & {"background", "occlusion"} else None
    for j, kind in enumerate(kinds):
        aug = augment(fs, kind, seed=seed + 1 + j, obj_mask=obj_mask)
        parts.append(DS.build_beam_dataset(aug, representation, seed=seed + 101 + j, **kw))

    return {
        "X": np.concatenate([p["X"] for p in parts]),
        "y": np.concatenate([p["y"] for p in parts]),
        "meta": pd.concat([p["meta"] for p in parts], ignore_index=True),
        "feature_names": parts[0]["feature_names"],
        "representation": representation,
        "kinds": list(kinds),
    }


# ---------------------------------------------------------------------------
# Leg 3: where in the cutout does the model look?
# ---------------------------------------------------------------------------
def cutout_region_mask(
    n_samples: int = config.CUTOUT_N_SAMPLES,
    window_m: float = config.CUTOUT_WIDTH_M,
    core_half_m: float = config.OBJECT_CORE_HALF_WIDTH_M,
) -> np.ndarray:
    """True for cutout samples inside the object core, False for the flanks.

    Only meaningful because the cutout is distance-normalised: sample k sits at
    the same physical lateral offset regardless of range, so "the core" is a
    fixed physical region rather than a fixed number of beams.
    """
    return np.abs(FE.cutout_lateral_offsets(n_samples, window_m)) <= core_half_m


def importance_by_region(model, representation: str = "cutout") -> pd.DataFrame:
    """Random-forest importance mass in the object core versus the background flanks."""
    if representation != "cutout":
        raise ValueError("region attribution is defined for the cutout representation")
    core = cutout_region_mask()
    imp = model.feature_importances_
    return pd.DataFrame(
        {
            "region": ["object core", "background flanks"],
            "n_features": [int(core.sum()), int((~core).sum())],
            "importance_mass": [float(imp[core].sum()), float(imp[~core].sum())],
            "importance_per_feature": [
                float(imp[core].mean()), float(imp[~core].mean())
            ],
        }
    )


def blank_flanks(X: np.ndarray, fill: float = 1.0) -> np.ndarray:
    """Replace the background flanks of every cutout with a constant.

    Applied at INFERENCE to an already-trained model. It answers a narrow,
    decisive question: does this model's test-time behaviour depend on the
    background at all? If test AP is unchanged, the background is not what the
    model is using and the hypothesis is refuted regardless of what the
    augmentation ablation says.
    """
    core = cutout_region_mask()
    Xb = X.copy()
    Xb[:, ~core] = fill
    return Xb
