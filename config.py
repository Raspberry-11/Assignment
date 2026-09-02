"""Central configuration for the DA331 DROW assignment.

Every path and physical constant used anywhere in the project lives here, so
that the notebook has exactly one cell to edit when moving between machines.
"""

from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# EDIT THIS ONE LINE when moving machines.
#   Windows example: DATA_ROOT = Path(r"C:\sem 5\big_data\assignments\assignment_2\DROWv2-data")
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data" / "DROWv2-data"

CACHE_DIR = PROJECT_ROOT / "cache"
FIGURE_DIR = PROJECT_ROOT / "figures"

SPLITS = ("train", "val", "test")

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
SEED = 331

# --------------------------------------------------------------------------
# Sensor geometry (SICK S300, per DROW v1 paper Section III.A)
# --------------------------------------------------------------------------
N_BEAMS = 450
FOV_DEG = 225.0
ANGULAR_RES_DEG = FOV_DEG / N_BEAMS  # 0.5 degrees
SCAN_HZ = 12.5
SENSOR_HEIGHT_M = 0.37

# Bearing phi is positive to the robot's left, 0 straight ahead, negative to the
# right. Bearing INCREASES with beam index, so beam 0 is the rightmost beam and
# beam 449 the leftmost.
#
# The assignment brief states the opposite ("index 0 is the leftmost beam"). That
# claim is tested against the annotations in notebook section B2, which scores
# both conventions by whether an annotated person lands on a beam whose measured
# range matches the annotated range. The increasing convention agrees 78% of the
# time on the exact beam; the decreasing one agrees 10%, which is chance. So the
# data wins and the brief is wrong on this point.
BEAM_ANGLES_RAD = np.linspace(-np.radians(FOV_DEG / 2.0), np.radians(FOV_DEG / 2.0), N_BEAMS)
BEAM_ANGLES_DEG = np.degrees(BEAM_ANGLES_RAD)

# Missing-return sentinel. Value asserted by the assignment brief as "roughly
# 29.96"; src.io.sentinel_audit() verifies the exact value against the bytes on
# disk rather than trusting this constant.
SENTINEL_NOMINAL = 29.96

# --------------------------------------------------------------------------
# On-disk schema
# --------------------------------------------------------------------------
# The upstream README claims each scan line is "sequence number followed by 450
# values" (451 fields). The files actually released carry 452 fields:
#     seq, time_seconds, r_0 ... r_449
# src.io.probe_schema() re-derives this per file; nothing downstream assumes it.
EXPECTED_SCAN_FIELDS = 452
LABEL_EXTENSIONS = {"wc": "wheelchair", "wa": "walking_aid", "wp": "person"}

# The class this study commits to. Others appear in label EDA for contrast only.
TARGET_CLASS = "wp"

# --------------------------------------------------------------------------
# Spark
# --------------------------------------------------------------------------
SPARK_DRIVER_MEMORY = "4g"
SPARK_SHUFFLE_PARTITIONS = 16

# --------------------------------------------------------------------------
# Labelling (Part B/C)
# --------------------------------------------------------------------------
# A beam counts as "on a person" if its measured endpoint falls within this
# radius of an annotated centre. Chosen from the sensitivity sweep in
# src.labels.radius_sensitivity(), not from the literature; the notebook shows
# the curve and the plateau it sits on.
PERSON_RADIUS_M = 0.35

# Physical width used for the w/(r*dtheta) beams-on-object prediction.
PERSON_WIDTH_M = 0.5

# Candidate beams are restricted to plausible returns. Below 0.35 m the S300 is
# reading the platform itself; beyond 15 m a person is under one beam wide.
CANDIDATE_MIN_RANGE_M = 0.35
CANDIDATE_MAX_RANGE_M = 15.0

# --------------------------------------------------------------------------
# Cutout representation (Part C)
# --------------------------------------------------------------------------
CUTOUT_N_SAMPLES = 48        # resampled points per cutout
CUTOUT_WIDTH_M = 1.0         # physical width the angular window subtends
CUTOUT_DEPTH_M = 1.0         # radial clip, in metres, about the centre range
CUTOUT_MIN_HALF_BEAMS = 3.0  # floor, so a far-away window is not degenerate
CUTOUT_MAX_HALF_BEAMS = 90.0 # ceiling, so a 0.4 m reading does not eat the scan
RAW_WINDOW_HALF_BEAMS = 20   # fixed-window control representation

# --------------------------------------------------------------------------
# Modelling (Part D)
# --------------------------------------------------------------------------
NEGATIVE_RATIO = 10          # negatives kept per positive beam, in TRAINING only
TRAIN_FRAME_STRIDE = 2       # keep every Nth annotated train frame
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 16
RF_MIN_SAMPLES_LEAF = 5
RF_N_JOBS = -1

# Detection post-processing and matching.
NMS_RADIUS_M = 0.5
MATCH_RADIUS_M = 0.5
RANGE_BANDS = ((0.0, 2.0), (2.0, 4.0), (4.0, 7.0), (7.0, 15.0))

# --------------------------------------------------------------------------
# Part E
# --------------------------------------------------------------------------
# Radial offset beyond which a cutout sample is treated as background context
# rather than as part of the object it is centred on.
OBJECT_CORE_HALF_WIDTH_M = 0.35
AUG_RANGE_NOISE_SD_M = 0.03      # SICK S300 systematic error is ~30 mm
AUG_BG_PERTURB_SD_M = 0.30       # corridor-geometry perturbation
AUG_DROPOUT_P = 0.10             # per-beam occlusion dropout probability
