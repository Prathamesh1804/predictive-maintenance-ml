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

# Project root derived portably from source file location
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_candidate_data_dirs(specified_dir: Optional[Path] = None) -> list[Path]:
    """
    Return an ordered list of candidate directories where raw CSVs may reside.
    Avoids reliance on current working directory while checking all common locations.
    """
    candidates = []
    if specified_dir is not None:
        candidates.append(Path(specified_dir))

    candidates.extend([
        PROJECT_ROOT / "data" / "raw",
        PROJECT_ROOT / "data",
        RAW_DIR,
        Path.cwd() / "data" / "raw",
        Path.cwd() / "data",
        Path.cwd(),
    ])

    seen = set()
    unique = []
    for c in candidates:
        try:
            res = c.resolve()
            if res not in seen:
                seen.add(res)
                unique.append(c)
        except Exception:
            if c not in seen:
                seen.add(c)
                unique.append(c)
    return unique


def download_dataset(raw_dir: Optional[Path] = None, target_filename: str = "ai4i2020.csv") -> Optional[Path]:
    """
    Attempt to download the official UCI AI4I 2020 dataset and save it to raw_dir.
    Returns Path to the downloaded CSV on success, or None on failure.
    """
    effective_dir = Path(raw_dir) if raw_dir is not None else (PROJECT_ROOT / "data" / "raw")
    try:
        import urllib.request
        import urllib.error
        import zipfile
        import io

        effective_dir.mkdir(parents=True, exist_ok=True)
        target_path = effective_dir / target_filename
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
        target_path = effective_dir / target_filename
        df.to_csv(target_path, index=False)
        if target_path.exists() and target_path.stat().st_size > 0:
            logger.info("Successfully fetched dataset via ucimlrepo: %s", target_path)
            return target_path
    except Exception as exc:
        logger.debug("ucimlrepo fetch unavailable or failed: %s", exc)

    return None


def find_raw_csv(raw_dir: Optional[Path] = None) -> Optional[Path]:
    """Scan candidate directories for any accepted CSV filename and return its path."""
    search_dirs = get_candidate_data_dirs(raw_dir)

    for d in search_dirs:
        if not d.exists() or not d.is_dir():
            continue

        # 1. Search for accepted candidate filenames (exact and case-insensitive)
        for name in RAW_CSV_CANDIDATES:
            exact = d / name
            if exact.exists() and exact.is_file():
                return exact

            try:
                for item in d.iterdir():
                    if item.is_file() and item.name.lower() == name.lower():
                        return item
            except Exception:
                pass

        # 2. Fallback: accept any .csv in candidate directory
        try:
            csv_files = [f for f in d.iterdir() if f.is_file() and f.suffix.lower() == ".csv"]
            if csv_files:
                logger.warning(
                    "No standard filename found in %s; using first CSV found: %s", d, csv_files[0].name
                )
                return csv_files[0]
        except Exception:
            pass

    return None


def load_raw_data(raw_dir: Optional[Path] = None, auto_download: bool = True) -> pd.DataFrame:
    """
    Load the raw UCI AI4I 2020 dataset CSV.
    Attempts automatic download from UCI if the dataset is missing.
    Ensures UTF-8 BOM is stripped and headers are normalised.

    Raises
    ------
    FileNotFoundError
        When no CSV is found and automatic download fails.
    ValueError
        When the CSV fails column or basic shape validation.
    """
    csv_path = find_raw_csv(raw_dir)
    effective_dir = Path(raw_dir) if raw_dir is not None else (PROJECT_ROOT / "data" / "raw")

    if csv_path is None and auto_download:
        csv_path = download_dataset(effective_dir)

    if csv_path is None:
        instructions = (
            "DATASET NOT FOUND:\n"
            f"  Expected location: {effective_dir.resolve() / 'ai4i2020.csv'}\n"
            "  The raw dataset CSV was not found in any standard project directory,\n"
            "  and automatic download could not be completed.\n\n"
            "To fix this in Streamlit Community Cloud:\n"
            "  1. Ensure 'data/raw/ai4i2020.csv' exists in the repository.\n"
            "  2. Verify it is tracked in Git with: git status\n"
            "  3. Commit and push: git push origin main"
        )
        raise FileNotFoundError(instructions)

    logger.info("Loading dataset from: %s", csv_path)

    # Use utf-8-sig to automatically strip UTF-8 Byte Order Mark (\xef\xbb\xbf)
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(csv_path, encoding="utf-8", errors="replace")
        except Exception:
            df = pd.read_csv(csv_path, encoding="latin1")

    # Clean BOM and strip whitespace from all column names
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]

    logger.info("Loaded %d rows × %d columns from %s", df.shape[0], df.shape[1], csv_path)
    return df


def validate_dataset(df: pd.DataFrame) -> dict:
    """
    Validate the raw dataset and return a validation report dict.

    Checks column presence, row count, data types, missing values, and
    target distribution. Does NOT raise on warnings — returns them in
    the report so the caller can decide how to handle them.
    """
    # Clean headers defensive check
    clean_columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    report: dict = {
        "row_count": len(df),
        "column_count": len(clean_columns),
        "columns_found": clean_columns,
        "missing_columns": [],
        "extra_columns": [],
        "missing_values": {},
        "target_distribution": {},
        "warnings": [],
        "errors": [],
        "valid": True,
    }

    # ── Column checks ─────────────────────────────────────────────────────────
    found = set(clean_columns)
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


def load_and_validate(raw_dir: Optional[Path] = None, auto_download: bool = True) -> tuple[pd.DataFrame, dict]:
    """Convenience: load the CSV and return (dataframe, validation_report)."""
    df = load_raw_data(raw_dir, auto_download=auto_download)
    report = validate_dataset(df)
    if not report["valid"]:
        raise ValueError(
            "Dataset failed validation:\n" + "\n".join(report["errors"])
        )
    return df, report
