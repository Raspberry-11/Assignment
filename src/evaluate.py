"""Part D: from per-beam scores to detections, and from detections to honest numbers.

The metric is average precision under a distance-based match criterion, which
is the only thing that makes sense here. Per-beam accuracy would be 99.5%
before any modelling, because 99.5% of beams are wall. Per-beam AUC would be
better but still answers the wrong question: the task is "how many people did
you find and how many did you invent", and that is a detection metric.

Match criterion: a detection matches a ground-truth person if it lands within
MATCH_RADIUS_M in the plane. Matching is greedy in descending score, one
ground truth per detection and one detection per ground truth, which makes
duplicate detections false positives rather than free recall.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src import features as FE  # noqa: E402
from src import labels as L  # noqa: E402


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
def random_forest(seed: int = config.SEED, **kw) -> RandomForestClassifier:
    """The one model this study commits to.

    A forest is chosen over a 1-D CNN deliberately: it trains on a laptop in
    under a minute, its feature importances are directly interpretable in the
    cutout's physical coordinates (which Part E depends on), and the assignment
    states outright that with good features it scores comparably here.
    """
    params = dict(
        n_estimators=config.RF_N_ESTIMATORS,
        max_depth=config.RF_MAX_DEPTH,
        min_samples_leaf=config.RF_MIN_SAMPLES_LEAF,
        n_jobs=config.RF_N_JOBS,
        class_weight="balanced_subsample",
        random_state=seed,
    )
    params.update(kw)
    return RandomForestClassifier(**params)


# ---------------------------------------------------------------------------
# Ground truth and the surface/centre offset
# ---------------------------------------------------------------------------
def ground_truth_table(fs) -> pd.DataFrame:
    """One row per annotated person: frame index, r, phi, x, y."""
    rows = []
    for i, objs in enumerate(fs.objects):
        for r, phi in objs:
            x, y = L.polar_to_cartesian(r, phi)
            rows.append((i, float(r), float(phi), float(x), float(y)))
    return pd.DataFrame(rows, columns=["frame", "r", "phi", "x", "y"])


def estimate_surface_offset(fs, radius_m: float = config.PERSON_RADIUS_M) -> float:
    """Median gap between an annotated centre and the nearest measured point.

    An annotator clicks the centre of a person; a laser only ever sees the
    front surface. A detection placed at a beam endpoint is therefore
    systematically nearer to the sensor than the label it must match. This
    measures that bias on TRAIN data so the detection can be pushed back by it,
    instead of silently spending a third of the 0.5 m match budget on it.
    """
    gaps = []
    for i, objs in enumerate(fs.objects):
        if not len(objs):
            continue
        bx, by = L.scan_to_cartesian(fs.R[i])
        v = fs.valid[i]
        if not v.any():
            continue
        for r, phi in objs:
            ox, oy = L.polar_to_cartesian(r, phi)
            d = np.hypot(bx[v] - ox, by[v] - oy)
            j = int(np.argmin(d))
            if d[j] <= radius_m:
                gaps.append(r - fs.R[i][v][j])
    return float(np.median(gaps)) if gaps else 0.0


# ---------------------------------------------------------------------------
# Scores -> detections
# ---------------------------------------------------------------------------
def nms_frame(
    beams: np.ndarray,
    scores: np.ndarray,
    ranges: np.ndarray,
    radius_m: float = config.NMS_RADIUS_M,
    min_score: float = 0.0,
    offset_m: float = 0.0,
):
    """Greedy non-maximum suppression in the plane, for one frame.

    Suppression is by Cartesian distance, not by beam index. Two beams 20 apart
    are 0.9 m apart at 5 m but 0.09 m apart at 0.5 m; an index-space radius
    would merge two people standing close by at short range and fail to merge
    one person at long range.
    """
    keep = scores > min_score
    beams, scores = beams[keep], scores[keep]
    if beams.size == 0:
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0, dtype=int)

    r = ranges[beams] + offset_m
    phi = L.beam_phi(beams)
    x, y = L.polar_to_cartesian(r, phi)

    # Vectorised greedy NMS. The loop runs once per SURVIVING detection (a few dozen
    # per frame) rather than once per candidate beam (a few hundred), and each
    # iteration suppresses the rest of the frame in a single numpy operation. The
    # naive nested-loop version costs ~100x more and is what makes a full-corpus
    # evaluation take hours instead of seconds.
    order = np.argsort(-scores)
    xs, ys = x[order], y[order]
    alive = np.ones(order.size, dtype=bool)
    kept_local = []
    r2 = radius_m ** 2
    for j in range(order.size):
        if not alive[j]:
            continue
        kept_local.append(j)
        alive &= (xs[j] - xs) ** 2 + (ys[j] - ys) ** 2 > r2
    kept = order[np.asarray(kept_local, dtype=int)]
    return np.stack([x[kept], y[kept]], axis=1), scores[kept], beams[kept]


def detections_from_beam_scores(
    fs,
    meta: pd.DataFrame,
    scores: np.ndarray,
    radius_m: float = config.NMS_RADIUS_M,
    min_score: float = 0.0,
    offset_m: float = 0.0,
) -> pd.DataFrame:
    """Run NMS per frame over a flat (meta, score) table of beam scores."""
    out = []
    frame_arr = meta["frame"].to_numpy()
    beam_arr = meta["beam"].to_numpy()
    order = np.argsort(frame_arr, kind="stable")
    fa, ba, sa = frame_arr[order], beam_arr[order], scores[order]
    bounds = np.searchsorted(fa, np.arange(len(fs) + 1))

    for i in range(len(fs)):
        lo, hi = bounds[i], bounds[i + 1]
        if hi <= lo:
            continue
        xy, sc, bm = nms_frame(ba[lo:hi], sa[lo:hi], fs.R[i], radius_m, min_score, offset_m)
        for (x, y), s, b in zip(xy, sc, bm):
            out.append((i, int(b), float(s), float(x), float(y), float(fs.R[i][b])))
    return pd.DataFrame(out, columns=["frame", "beam", "score", "x", "y", "r"])


# ---------------------------------------------------------------------------
# Matching and PR
# ---------------------------------------------------------------------------
def match_detections(
    det: pd.DataFrame, gt: pd.DataFrame, match_radius_m: float = config.MATCH_RADIUS_M
) -> pd.DataFrame:
    """Greedy one-to-one matching, descending score. Adds tp / gt_index / gt_r.

    Greedy-by-score rather than optimal assignment: it is what every detection
    benchmark does, it is what a downstream consumer of the detector would
    experience, and it never rewards a low-confidence detection by reassigning
    a ground truth away from a confident one.
    """
    det = det.sort_values("score", ascending=False).reset_index(drop=True)
    tp = np.zeros(len(det), dtype=bool)
    gt_index = np.full(len(det), -1, dtype=int)
    gt_r = np.full(len(det), np.nan)

    gt_by_frame = {
        f: (g.index.to_numpy(), g[["x", "y"]].to_numpy(), g["r"].to_numpy())
        for f, g in gt.groupby("frame")
    }
    taken = {f: np.zeros(len(idx), dtype=bool) for f, (idx, _, _) in gt_by_frame.items()}

    for k, row in enumerate(det.itertuples(index=False)):
        entry = gt_by_frame.get(row.frame)
        if entry is None:
            continue
        idx, xy, rr = entry
        free = ~taken[row.frame]
        if not free.any():
            continue
        d = np.hypot(xy[:, 0] - row.x, xy[:, 1] - row.y)
        d = np.where(free, d, np.inf)
        j = int(np.argmin(d))
        if d[j] <= match_radius_m:
            tp[k] = True
            taken[row.frame][j] = True
            gt_index[k] = idx[j]
            gt_r[k] = rr[j]

    det = det.copy()
    det["tp"], det["gt_index"], det["gt_r"] = tp, gt_index, gt_r
    return det


def pr_curve(det: pd.DataFrame, n_gt: int):
    """Precision/recall at every score threshold, from a matched detection table."""
    if len(det) == 0 or n_gt == 0:
        return np.array([1.0]), np.array([0.0]), np.array([np.inf])
    d = det.sort_values("score", ascending=False)
    tp = d["tp"].to_numpy().astype(float)
    ctp, cfp = np.cumsum(tp), np.cumsum(1.0 - tp)
    precision = ctp / np.maximum(ctp + cfp, 1e-12)
    recall = ctp / n_gt
    return precision, recall, d["score"].to_numpy()


def average_precision(precision: np.ndarray, recall: np.ndarray) -> float:
    """Area under the PR curve, summed over recall increments (no interpolation).

    Interpolated AP flatters a jagged curve. Since the whole point of this
    section is honest reporting, the un-interpolated version is used and said
    so out loud.
    """
    r = np.concatenate([[0.0], recall])
    return float(np.sum(np.diff(r) * precision))


def peak_f1(precision: np.ndarray, recall: np.ndarray, thresholds: np.ndarray) -> dict:
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    k = int(np.argmax(f1))
    return {
        "f1": float(f1[k]),
        "precision": float(precision[k]),
        "recall": float(recall[k]),
        "threshold": float(thresholds[k]),
    }


def evaluate_detections(det: pd.DataFrame, gt: pd.DataFrame, match_radius_m=config.MATCH_RADIUS_M) -> dict:
    matched = match_detections(det, gt, match_radius_m)
    p, r, t = pr_curve(matched, len(gt))
    return {
        "matched": matched,
        "precision": p,
        "recall": r,
        "thresholds": t,
        "ap": average_precision(p, r),
        "n_gt": len(gt),
        "n_det": len(det),
        "peak_f1": peak_f1(p, r, t),
    }


def range_band_report(
    result: dict, bands=config.RANGE_BANDS, width_m: float = config.PERSON_WIDTH_M
) -> pd.DataFrame:
    """AP per range band, printed next to the beams-on-object prediction.

    The two columns belong side by side: the reason AP collapses in the far
    band is that the object has stopped being resolvable, not that the
    classifier has stopped trying.
    """
    matched, gt = result["matched"], result["gt"]
    rows = []
    for lo, hi in bands:
        gt_b = gt[(gt["r"] >= lo) & (gt["r"] < hi)]
        # a detection belongs to the band of the object it matched; an
        # unmatched detection belongs to the band it was fired in
        band_r = np.where(matched["tp"], matched["gt_r"], matched["r"])
        det_b = matched[(band_r >= lo) & (band_r < hi)]
        p, r, _ = pr_curve(det_b, len(gt_b))
        mid = (lo + hi) / 2.0
        rows.append(
            {
                "band_m": f"{lo:g}-{hi:g}",
                "n_gt": len(gt_b),
                "n_det": len(det_b),
                "ap": average_precision(p, r) if len(gt_b) else np.nan,
                "max_recall": float(r[-1]) if len(r) else 0.0,
                "beams_on_person_at_mid": float(L.expected_beams_on_object(width_m, mid)),
            }
        )
    return pd.DataFrame(rows)


def run_evaluation(fs, meta, scores, offset_m=0.0, **kw) -> dict:
    """Beam scores -> detections -> matched -> AP, with the ground truth attached."""
    gt = ground_truth_table(fs)
    det = detections_from_beam_scores(fs, meta, scores, offset_m=offset_m, **kw)
    out = evaluate_detections(det, gt)
    out["gt"] = gt
    out["detections"] = det
    return out


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def baseline_all_negative(fs) -> dict:
    """Predict nothing, ever. AP is exactly 0 and recall is exactly 0.

    Worth printing rather than assuming, because it is the score that a
    per-beam accuracy of 99.5% corresponds to, and that contrast is the whole
    argument for not using accuracy.
    """
    gt = ground_truth_table(fs)
    return {
        "name": "all-negative",
        "ap": 0.0,
        "n_gt": len(gt),
        "n_det": 0,
        "peak_f1": {"f1": 0.0, "precision": 0.0, "recall": 0.0, "threshold": np.inf},
        "beam_accuracy_if_all_negative": None,
    }


def baseline_random_scores(meta: pd.DataFrame, seed: int = config.SEED) -> np.ndarray:
    """Uniform random score per candidate beam. Sets the floor a PR curve must beat."""
    return np.random.default_rng(seed).random(len(meta))


def geometric_rule_detections(
    fs,
    width_mu: float = 0.45,
    width_sigma: float = 0.25,
    min_points: int = 2,
    max_depth_m: float = 0.6,
    offset_m: float = 0.0,
) -> pd.DataFrame:
    """Jump-distance segmentation, scored by how person-shaped each segment is.

    This is the "simple geometric rule" baseline, and it is a real one: it is
    roughly what a 2005-era leg detector did before learning was involved. The
    score is a Gaussian on segment width so that it produces a PR curve rather
    than a single operating point, which makes it comparable to the forest.
    """
    rows = []
    for i in range(len(fs)):
        segs = FE.segment_scan(fs.R[i], fs.valid[i])
        desc = FE.segment_descriptors(fs.R[i], segs)
        if not len(desc):
            continue
        npts, width, depth, rc, phic = desc.T
        ok = (
            (npts >= min_points)
            & (depth <= max_depth_m)
            & (rc >= config.CANDIDATE_MIN_RANGE_M)
            & (rc <= config.CANDIDATE_MAX_RANGE_M)
        )
        score = np.exp(-0.5 * ((width - width_mu) / width_sigma) ** 2)
        for j in np.flatnonzero(ok):
            x, y = L.polar_to_cartesian(rc[j] + offset_m, phic[j])
            rows.append((i, -1, float(score[j]), float(x), float(y), float(rc[j])))
    return pd.DataFrame(rows, columns=["frame", "beam", "score", "x", "y", "r"])


# ---------------------------------------------------------------------------
# Velocity: can this run online at 12.5 Hz?
# ---------------------------------------------------------------------------
def latency_report(model, fs, representation: str = "cutout", n_frames: int = 200) -> pd.DataFrame:
    """Wall-clock per stage for a single frame, against the 80 ms scan budget.

    Measured, not estimated. A method that cannot keep up with its own sensor
    is a different method from one that can, and the brief asks for the number.
    """
    n = min(n_frames, len(fs))
    stages, t = {}, time.perf_counter
    t0 = t()
    for i in range(n):
        FE.missing_mask(fs.R[i], [config.SENTINEL_NOMINAL])
    stages["sentinel_mask"] = (t() - t0) / n
    t0 = t()
    Xs = [FE.extract(representation, fs.R[i], np.flatnonzero(fs.valid[i]), fs.missing[i]) for i in range(n)]
    stages["feature_extraction"] = (t() - t0) / n
    t0 = t()
    scores = [model.predict_proba(X)[:, 1] for X in Xs]
    stages["forest_inference"] = (t() - t0) / n
    t0 = t()
    for i in range(n):
        nms_frame(np.flatnonzero(fs.valid[i]), scores[i], fs.R[i])
    stages["nms"] = (t() - t0) / n

    df = pd.DataFrame({"stage": list(stages), "seconds_per_frame": list(stages.values())})
    df.loc[len(df)] = ["TOTAL", df["seconds_per_frame"].sum()]
    df["ms"] = df["seconds_per_frame"] * 1e3
    df["budget_frac_at_12.5Hz"] = df["seconds_per_frame"] * config.SCAN_HZ
    return df


def memory_report(model, fs, representation: str = "cutout") -> pd.DataFrame:
    """Steady-state memory an online version would hold. One frame plus one forest."""
    import pickle

    n_feat = len(FE.feature_names(representation))
    rows = [
        ("one scan frame (450 float64)", config.N_BEAMS * 8),
        ("candidate feature block (450 x %d float32)" % n_feat, config.N_BEAMS * n_feat * 4),
        ("serialised forest", len(pickle.dumps(model))),
    ]
    df = pd.DataFrame(rows, columns=["item", "bytes"])
    df.loc[len(df)] = ["TOTAL", df["bytes"].sum()]
    df["MB"] = df["bytes"] / 1e6
    return df


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------
def failure_frames(result: dict, score_threshold: float, n: int = 8) -> pd.DataFrame:
    """Frames ranked by how badly they go wrong at a chosen operating point.

    Returns false-positive and false-negative counts per frame so the notebook
    can plot the worst of each kind rather than a random sample, which is how
    you find a systematic failure instead of an anecdote.
    """
    matched = result["matched"]
    fired = matched[matched["score"] >= score_threshold]
    fp = fired[~fired["tp"]].groupby("frame").size().rename("false_positives")
    tp = fired[fired["tp"]].groupby("frame").size().rename("true_positives")
    ngt = result["gt"].groupby("frame").size().rename("n_gt")
    out = pd.concat([ngt, tp, fp], axis=1).fillna(0).astype(int)
    out["false_negatives"] = (out["n_gt"] - out["true_positives"]).clip(lower=0)
    out["errors"] = out["false_positives"] + out["false_negatives"]
    return out.sort_values("errors", ascending=False).head(n).reset_index()
