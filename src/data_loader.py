"""
Dataset loading, validation, and reporting for the UCI AI4I 2020
Predictive Maintenance Dataset.

NOTE: The UCI AI4I 2020 dataset is SYNTHETIC — it was generated to model
realistic industrial conditions but does not represent real machine telemetry.
Source: Matzka, S. (2020). AI4I 2020 Predictive Maintenance Dataset.
        UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C
"""
import pandas as pd
from pathlib import Path
from typing import Optional

from config.config import (
    RAW_DIR,
    RAW_CSV_CANDIDATES,
    LEAKAGE_COLUMNS,
    ID_COLUMNS,
    TARGET_COLUMN,
    BASE_FEATURES,
)
from src.utils import get_logger

logger = get_logger(__name__)

EXPECTED_COLUMNS = {
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
}

EXPECTED_ROWS = 10_000

# Verified official UCI ML Repository download URL
DATASET_URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"


def download_dataset(raw_dir: Path = RAW_DIR, target_filename: str = "ai4i2020.csv") -> Optional[Path]:
    """
    Attempt to download the official UCI AI4I 2020 dataset and save it to raw_dir.
    Returns Path to the downloaded CSV on success, or None on failure.
    """
    try:
        import urllib.request
        import urllib.error
        import zipfile
        import io

        raw_dir.mkdir(parents=True, exist_ok=True)
        target_path = raw_dir / target_filename
        if target_path.exists() and target_path.stat().st_size > 0:
            return target_path

        logger.info("Attempting automatic download from UCI ML Repository: %s", DATASET_URL)
        req = urllib.request.Request(
            DATASET_URL,
            headers={"User-Agent": "Mozilla/5.0 (PredictiveMaintenanceML/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            content = response.read()

        try:
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
                if not csv_names:
                    logger.warning("No CSV file found inside the downloaded ZIP archive.")
                    return None
                selected_name = next((n for n in csv_names if "ai4i" in n.lower()), csv_names[0])
                csv_data = zf.read(selected_name)
                with open(target_path, "wb") as f:
                    f.write(csv_data)
        except zipfile.BadZipFile:
            with open(target_path, "wb") as f:
                f.write(content)

        if target_path.exists() and target_path.stat().st_size > 0:
            logger.info("Successfully downloaded and saved dataset to: %s", target_path)
            return target_path
    except Exception as exc:
        logger.warning("Automatic dataset download from UCI failed: %s", exc)

    # Optional ucimlrepo package fallback
    try:
        from ucimlrepo import fetch_ucirepo
        logger.info("Attempting dataset fetch via ucimlrepo package...")
        dataset = fetch_ucirepo(id=601)
        df = dataset.data.original
        target_path = raw_dir / target_filename
        df.to_csv(target_path, index=False)
        if target_path.exists() and target_path.stat().st_size > 0:
            logger.info("Successfully fetched dataset via ucimlrepo: %s", target_path)
            return target_path
    except Exception as exc:
        logger.debug("ucimlrepo fetch unavailable or failed: %s", exc)

    return None


def find_raw_csv(raw_dir: Path = RAW_DIR) -> Optional[Path]:
    """Scan *raw_dir* for any accepted CSV filename and return its path."""
    for name in RAW_CSV_CANDIDATES:
        candidate = raw_dir / name
        if candidate.exists():
            return candidate
    # Also accept any .csv file in the raw directory as a fallback
    csv_files = list(raw_dir.glob("*.csv"))
    if csv_files:
        logger.warning(
            "No standard filename found; using first CSV found: %s", csv_files[0].name
        )
        return csv_files[0]
    return None


def load_raw_data(raw_dir: Path = RAW_DIR, auto_download: bool = True) -> pd.DataFrame:
    """
    Load the raw UCI AI4I 2020 dataset CSV.
    Attempts automatic download from UCI if the dataset is missing.

    Raises
    ------
    FileNotFoundError
        When no CSV is found in *raw_dir* and automatic download fails.
    ValueError
        When the CSV fails column or basic shape validation.
    """
    csv_path = find_raw_csv(raw_dir)
    if csv_path is None and auto_download:
        csv_path = download_dataset(raw_dir)

    if csv_path is None:
        instructions = (
            "\n"
            "=" * 55 + "\n"
            "  DATASET NOT FOUND - Manual download or git commit required\n"
            "=" * 55 + "\n"
            "  The UCI AI4I 2020 dataset was not found in:\n"
            f"     {raw_dir.resolve()}\n"
            "  and automatic download could not be completed.\n\n"
            "  To resolve this:\n"
            "  1. Download the dataset from UCI:\n"
            "     https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset\n"
            "  2. Extract 'ai4i2020.csv' and place it in:\n"
            f"     {raw_dir.resolve()}\n"
            "  3. If deploying to Streamlit Community Cloud, track and commit the file:\n"
            "     git add -f data/raw/ai4i2020.csv\n"
            "     git commit -m 'Add AI4I 2020 dataset'\n"
            "     git push\n"
            "=" * 55 + "\n"
        )
        raise FileNotFoundError(instructions)

    logger.info("Loading dataset from: %s", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d rows × %d columns", *df.shape)
    return df


def validate_dataset(df: pd.DataFrame) -> dict:
    """
    Validate the raw dataset and return a validation report dict.

    Checks column presence, row count, data types, missing values, and
    target distribution. Does NOT raise on warnings — returns them in
    the report so the caller can decide how to handle them.
    """
    report: dict = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns_found": list(df.columns),
        "missing_columns": [],
        "extra_columns": [],
        "missing_values": {},
        "target_distribution": {},
        "warnings": [],
        "errors": [],
        "valid": True,
    }

    # ── Column checks ─────────────────────────────────────────────────────────
    found = set(df.columns)
    missing = EXPECTED_COLUMNS - found
    extra = found - EXPECTED_COLUMNS
    if missing:
        report["missing_columns"] = sorted(missing)
        report["errors"].append(f"Missing columns: {sorted(missing)}")
        report["valid"] = False
    if extra:
        report["extra_columns"] = sorted(extra)
        report["warnings"].append(f"Unexpected extra columns: {sorted(extra)}")

    # ── Row count check ───────────────────────────────────────────────────────
    if len(df) < 100:
        report["errors"].append(f"Dataset has only {len(df)} rows — suspiciously small.")
        report["valid"] = False
    elif len(df) != EXPECTED_ROWS:
        report["warnings"].append(
            f"Expected {EXPECTED_ROWS} rows, got {len(df)}. "
            "Dataset may be a subset or augmented version."
        )

    # ── Missing values ────────────────────────────────────────────────────────
    mv = df.isnull().sum()
    mv = mv[mv > 0].to_dict()
    report["missing_values"] = {k: int(v) for k, v in mv.items()}
    if mv:
        report["warnings"].append(f"Missing values detected: {mv}")

    # ── Target distribution ───────────────────────────────────────────────────
    if TARGET_COLUMN in df.columns:
        vc = df[TARGET_COLUMN].value_counts().to_dict()
        report["target_distribution"] = {int(k): int(v) for k, v in vc.items()}
        failure_rate = vc.get(1, 0) / len(df) * 100
        report["failure_rate_pct"] = round(failure_rate, 4)
        if failure_rate < 1 or failure_rate > 20:
            report["warnings"].append(
                f"Unusual failure rate: {failure_rate:.2f}%. "
                "Confirm the target column is correct."
            )

    # ── Data type checks ──────────────────────────────────────────────────────
    numeric_cols = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]
    for col in numeric_cols:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            report["errors"].append(f"Column '{col}' expected numeric, got {df[col].dtype}")
            report["valid"] = False

    # ── Log summary ───────────────────────────────────────────────────────────
    if report["valid"]:
        logger.info("Dataset validation PASSED. %d rows, failure rate=%.2f%%",
                    len(df), report.get("failure_rate_pct", 0.0))
    else:
        logger.error("Dataset validation FAILED: %s", report["errors"])

    for w in report["warnings"]:
        logger.warning(w)

    return report


def load_and_validate(raw_dir: Path = RAW_DIR, auto_download: bool = True) -> tuple[pd.DataFrame, dict]:
    """Convenience: load the CSV and return (dataframe, validation_report)."""
    df = load_raw_data(raw_dir, auto_download=auto_download)
    report = validate_dataset(df)
    if not report["valid"]:
        raise ValueError(
            "Dataset failed validation:\n" + "\n".join(report["errors"])
        )
    return df, report
