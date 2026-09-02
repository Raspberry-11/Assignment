"""Part A: ingestion, validation and storage engineering.

Nothing in this module trusts a documented schema. Every structural claim made
by the assignment brief or the upstream README is re-derived from the bytes on
disk and returned as evidence the notebook can print.

Spark is used for the scan data (~10^8 range readings across 113 recordings),
which is the part of this dataset that genuinely does not fit a per-row Python
loop. Labels cover ~5% of frames and are handled in src/labels.py with plain
Python, which is the honest tool for their size.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------
def spark_session(app_name: str = "da331-drow") -> SparkSession:
    """Local Spark session sized for a laptop, not a cluster.

    Using local[*] is a deliberate choice, defended in the notebook: the volume
    here justifies a distributed *execution model* (lazy, partitioned, spilled)
    but not a distributed *cluster*.
    """
    return (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", config.SPARK_DRIVER_MEMORY)
        .config("spark.sql.shuffle.partitions", config.SPARK_SHUFFLE_PARTITIONS)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


# ---------------------------------------------------------------------------
# A1. File inventory
# ---------------------------------------------------------------------------
def inventory(data_root: Path | None = None) -> pd.DataFrame:
    """One row per recording base name, one boolean column per extension.

    Catches recordings whose file set is incomplete. The val split is known to
    contain at least one base name with label files but no scan file; this
    function is how that is discovered rather than assumed.
    """
    data_root = Path(data_root or config.DATA_ROOT)
    exts = ["csv", "wc", "wa", "wp", "odom2"]
    rows = {}

    for split in config.SPLITS:
        split_dir = data_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Split directory not found: {split_dir}. Check config.DATA_ROOT."
            )
        for ext in exts:
            for path in sorted(split_dir.glob(f"*.{ext}")):
                base = path.name[: -(len(ext) + 1)]
                key = (split, base)
                row = rows.setdefault(
                    key,
                    {"split": split, "basename": base, **{e: False for e in exts},
                     **{f"{e}_bytes": 0 for e in exts}},
                )
                row[ext] = True
                row[f"{ext}_bytes"] = path.stat().st_size

    df = pd.DataFrame(rows.values()).sort_values(["split", "basename"])
    df["complete"] = df[exts].all(axis=1)
    return df.reset_index(drop=True)


def inventory_summary(inv: pd.DataFrame) -> pd.DataFrame:
    """Per-split counts, plus a count of incomplete recordings."""
    exts = ["csv", "wc", "wa", "wp", "odom2"]
    agg = inv.groupby("split")[exts].sum()
    agg["recordings"] = inv.groupby("split").size()
    agg["incomplete"] = inv.groupby("split")["complete"].apply(lambda s: (~s).sum())
    return agg.loc[list(config.SPLITS)]


def orphan_recordings(inv: pd.DataFrame) -> pd.DataFrame:
    """Recordings that are missing at least one of the five expected files."""
    return inv.loc[~inv["complete"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# A2. Schema forensics
# ---------------------------------------------------------------------------
def probe_schema(spark: SparkSession, data_root: Path | None = None) -> pd.DataFrame:
    """Count comma-separated fields on every line of every scan file.

    The upstream README describes 451 fields (seq + 450 ranges). This function
    exists because that description is wrong for the released files, and
    because the failure mode of trusting it is silent: a one-column offset
    shifts the entire angular axis without raising anything.

    Returns one row per file with the distinct field counts observed, so a
    ragged file would show up rather than being averaged away.
    """
    data_root = Path(data_root or config.DATA_ROOT)
    pattern = str(data_root / "*" / "*.csv")

    text = (
        spark.read.text(pattern)
        .withColumn("path", F.input_file_name())
        .withColumn("n_fields", F.size(F.split(F.col("value"), ",")))
    )
    per_file = (
        text.groupBy("path")
        .agg(
            F.count("*").alias("n_lines"),
            F.min("n_fields").alias("min_fields"),
            F.max("n_fields").alias("max_fields"),
            F.countDistinct("n_fields").alias("distinct_field_counts"),
        )
        .toPandas()
    )

    per_file["split"] = per_file["path"].str.extract(r"/(train|val|test)/")
    per_file["basename"] = per_file["path"].str.replace(r".*/", "", regex=True)
    per_file["ragged"] = per_file["distinct_field_counts"] > 1
    per_file["matches_readme_451"] = per_file["max_fields"] == 451
    per_file["matches_observed_452"] = per_file["max_fields"] == 452
    cols = ["split", "basename", "n_lines", "min_fields", "max_fields",
            "ragged", "matches_readme_451", "matches_observed_452"]
    return per_file[cols].sort_values(["split", "basename"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# A3. Typed scan loading
# ---------------------------------------------------------------------------
def load_scans(
    spark: SparkSession,
    data_root: Path | None = None,
    splits: tuple[str, ...] = config.SPLITS,
) -> DataFrame:
    """Parse scan CSVs into (split, file_id, seq, time_s, ranges[450]).

    The leading two fields are seq and a relative timestamp in seconds. The
    timestamp is NOT a range reading; including it as beam 0 is the specific
    bug this parser is written to prevent. It is retained as its own column
    because it is the only reliable join key to the odometry files.
    """
    data_root = Path(data_root or config.DATA_ROOT)
    paths = [str(data_root / s / "*.csv") for s in splits]

    text = (
        spark.read.text(paths)
        .withColumn("path", F.input_file_name())
        .withColumn("parts", F.split(F.col("value"), ","))
    )

    n = config.EXPECTED_SCAN_FIELDS
    bad = text.filter(F.size("parts") != n).limit(1).count()
    if bad:
        raise ValueError(
            f"Found scan lines whose field count != {n}. Run probe_schema() and "
            "reconcile before parsing; do not silently coerce."
        )

    ranges = F.transform(
        F.slice(F.col("parts"), 3, config.N_BEAMS),
        lambda x: x.cast(T.FloatType()),
    )
    return (
        text.withColumn("split", F.regexp_extract("path", r"/(train|val|test)/", 1))
        .withColumn("file_id", F.regexp_replace(F.regexp_extract("path", r"([^/]+)\.csv$", 1), r"\.bag$", ""))
        .withColumn("seq", F.col("parts").getItem(0).cast(T.IntegerType()))
        .withColumn("time_s", F.col("parts").getItem(1).cast(T.DoubleType()))
        .withColumn("ranges", ranges)
        .select("split", "file_id", "seq", "time_s", "ranges")
    )


def load_odometry(
    spark: SparkSession,
    data_root: Path | None = None,
    splits: tuple[str, ...] = config.SPLITS,
) -> DataFrame:
    """Parse .odom2 into (split, file_id, odom_seq, time_s, tx, ty, phi).

    The sequence column is named odom_seq rather than seq on purpose. It does
    NOT share a numbering space with the scan sequence numbers; joining the two
    on seq produces a near-empty result. Join on time_s. See
    verify_odom_join_key() for the evidence behind that claim.

    Values are relative to an arbitrary origin; only differences are meaningful.
    """
    data_root = Path(data_root or config.DATA_ROOT)
    paths = [str(data_root / s / "*.odom2") for s in splits]

    text = (
        spark.read.text(paths)
        .withColumn("path", F.input_file_name())
        .withColumn("parts", F.split(F.col("value"), ","))
    )
    return (
        text.withColumn("split", F.regexp_extract("path", r"/(train|val|test)/", 1))
        .withColumn("file_id", F.regexp_replace(F.regexp_extract("path", r"([^/]+)\.odom2$", 1), r"\.bag$", ""))
        .withColumn("odom_seq", F.col("parts").getItem(0).cast(T.IntegerType()))
        .withColumn("time_s", F.col("parts").getItem(1).cast(T.DoubleType()))
        .withColumn("tx", F.col("parts").getItem(2).cast(T.DoubleType()))
        .withColumn("ty", F.col("parts").getItem(3).cast(T.DoubleType()))
        .withColumn("phi", F.col("parts").getItem(4).cast(T.DoubleType()))
        .select("split", "file_id", "odom_seq", "time_s", "tx", "ty", "phi")
    )


# ---------------------------------------------------------------------------
# A4. Validation checks
# ---------------------------------------------------------------------------
def sentinel_audit(scans: DataFrame, top_n: int = 12) -> pd.DataFrame:
    """Most frequent range values, to identify missing-value encodings.

    The brief says the sentinel is "approximately 29.96" and asks whether it is
    the only one. Rounding to 2dp and counting is how that gets answered.
    A `> 29` filter is the obvious wrong move: genuine long returns live there.
    """
    return (
        scans.select(F.explode("ranges").alias("r"))
        .withColumn("r2", F.round("r", 2))
        .groupBy("r2")
        .count()
        .orderBy(F.desc("count"))
        .limit(top_n)
        .toPandas()
    )


def sentinel_neighbourhood(scans: DataFrame, low: float = 29.0) -> pd.DataFrame:
    """Distinct values above `low`, to separate the sentinel from real returns."""
    return (
        scans.select(F.explode("ranges").alias("r"))
        .filter(F.col("r") > low)
        .withColumn("r3", F.round("r", 3))
        .groupBy("r3")
        .count()
        .orderBy(F.desc("count"))
        .toPandas()
    )


def scan_integrity(scans: DataFrame) -> pd.DataFrame:
    """Per-file frame count, seq range, duplicate seqs, and implied frame rate.

    The implied rate is compared against the documented 12.5 Hz. A file whose
    implied rate is far off is either dropping frames or has a broken clock,
    and either way should not be pooled blindly with the rest.
    """
    return (
        scans.groupBy("split", "file_id")
        .agg(
            F.count("*").alias("n_frames"),
            F.countDistinct("seq").alias("n_distinct_seq"),
            F.min("seq").alias("seq_min"),
            F.max("seq").alias("seq_max"),
            F.min("time_s").alias("t_min"),
            F.max("time_s").alias("t_max"),
        )
        .withColumn("duplicate_seqs", F.col("n_frames") - F.col("n_distinct_seq"))
        .withColumn("duration_s", F.col("t_max") - F.col("t_min"))
        .withColumn("implied_hz", F.col("n_frames") / F.col("duration_s"))
        .withColumn("seq_span", F.col("seq_max") - F.col("seq_min") + 1)
        .withColumn("dropped_frames", F.col("seq_span") - F.col("n_frames"))
        .toPandas()
        .sort_values(["split", "file_id"])
        .reset_index(drop=True)
    )


def verify_odom_join_key(scans: DataFrame, odom: DataFrame) -> pd.DataFrame:
    """Evidence that .odom2 must be joined on time, not on sequence number.

    For each recording, reports how many scan seqs also appear as odom seqs,
    against how many rows each file has. If the sequence spaces were shared we
    would expect near-total overlap. They are not.
    """
    s = scans.groupBy("split", "file_id").agg(
        F.count("*").alias("scan_rows"),
        F.min("seq").alias("scan_seq_min"),
        F.max("seq").alias("scan_seq_max"),
    )
    o = odom.groupBy("split", "file_id").agg(
        F.count("*").alias("odom_rows"),
        F.min("odom_seq").alias("odom_seq_min"),
        F.max("odom_seq").alias("odom_seq_max"),
    )
    overlap = (
        scans.select("split", "file_id", F.col("seq").alias("k"))
        .join(odom.select("split", "file_id", F.col("odom_seq").alias("k")),
              on=["split", "file_id", "k"], how="inner")
        .groupBy("split", "file_id")
        .agg(F.count("*").alias("seq_overlap"))
    )
    out = (
        s.join(o, on=["split", "file_id"], how="outer")
        .join(overlap, on=["split", "file_id"], how="left")
        .fillna({"seq_overlap": 0})
        .withColumn("rows_match", F.col("scan_rows") == F.col("odom_rows"))
        .withColumn("seq_overlap_frac",
                    F.col("seq_overlap") / F.greatest(F.col("scan_rows"), F.lit(1)))
        .toPandas()
    )
    return out.sort_values(["split", "file_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# A5. Storage engineering
# ---------------------------------------------------------------------------
def build_cache(
    scans: DataFrame,
    odom: DataFrame,
    cache_dir: Path | None = None,
) -> dict:
    """Write the parsed data to partitioned Parquet and time the round trip.

    Returns the measurements the notebook needs to make the storage-format
    argument quantitatively rather than by assertion: bytes on disk and seconds
    to load, CSV versus Parquet.
    """
    cache_dir = Path(cache_dir or config.CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stats = {}

    for name, df in (("scans", scans), ("odom", odom)):
        target = cache_dir / name
        t0 = time.perf_counter()
        df.write.mode("overwrite").partitionBy("split").parquet(str(target))
        stats[f"{name}_write_s"] = time.perf_counter() - t0
        stats[f"{name}_parquet_bytes"] = sum(
            p.stat().st_size for p in target.rglob("*.parquet")
        )
    return stats


def read_cache(spark: SparkSession, name: str, cache_dir: Path | None = None) -> DataFrame:
    cache_dir = Path(cache_dir or config.CACHE_DIR)
    return spark.read.parquet(str(cache_dir / name))


def dir_bytes(path: Path, pattern: str) -> int:
    return sum(p.stat().st_size for p in Path(path).rglob(pattern))
