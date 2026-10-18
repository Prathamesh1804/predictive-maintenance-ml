"""
Download script for the UCI AI4I 2020 Predictive Maintenance Dataset.

Attempts to download directly from UCI ML Repository API.
If the download fails, prints clear manual instructions.
"""
import sys
import zipfile
import io
from pathlib import Path

# Ensure project root is on the path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.config import RAW_DIR
from src.utils import get_logger, ensure_dirs

logger = get_logger("download_data")

# UCI ML Repository direct download (ucimlrepo package or direct URL)
DATASET_URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
TARGET_FILENAME = "ai4i2020.csv"


def download_from_uci() -> bool:
    """
    Attempt to download the dataset ZIP from UCI and extract the CSV.
    Returns True on success, False on failure.
    """
    try:
        import urllib.request
        import urllib.error

        logger.info("Attempting to download from UCI ML Repository...")
        logger.info("URL: %s", DATASET_URL)

        ensure_dirs(RAW_DIR)
        target_path = RAW_DIR / TARGET_FILENAME

        if target_path.exists():
            logger.info("Dataset already exists at: %s", target_path)
            return True

        # Set a reasonable timeout
        req = urllib.request.Request(
            DATASET_URL,
            headers={"User-Agent": "Mozilla/5.0 (educational use)"},
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            content = response.read()

        # Try to extract as ZIP
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
                if not csv_names:
                    logger.error("No CSV found inside the downloaded ZIP.")
                    return False
                # Use the first CSV found
                csv_data = zf.read(csv_names[0])
                with open(target_path, "wb") as f:
                    f.write(csv_data)
            logger.info("[OK] Dataset extracted to: %s", target_path)
            return True
        except zipfile.BadZipFile:
            # Maybe the response IS the CSV directly
            with open(target_path, "wb") as f:
                f.write(content)
            logger.info("[OK] Dataset saved to: %s", target_path)
            return True

    except Exception as exc:
        logger.warning("Automatic download failed: %s", exc)
        return False


def try_ucimlrepo() -> bool:
    """
    Try using the `ucimlrepo` package if installed.
    """
    try:
        from ucimlrepo import fetch_ucirepo

        logger.info("Attempting download via ucimlrepo package...")
        dataset = fetch_ucirepo(id=601)
        df = dataset.data.original

        ensure_dirs(RAW_DIR)
        target_path = RAW_DIR / TARGET_FILENAME
        df.to_csv(target_path, index=False)
        logger.info("[OK] Dataset saved via ucimlrepo: %s", target_path)
        return True
    except ImportError:
        logger.info("ucimlrepo package not installed (optional).")
        return False
    except Exception as exc:
        logger.warning("ucimlrepo download failed: %s", exc)
        return False


def print_manual_instructions():
    """Print step-by-step manual download instructions."""
    print("\n" + "═" * 65)
    print("  MANUAL DATASET DOWNLOAD INSTRUCTIONS")
    print("═" * 65)
    print()
    print("  The UCI AI4I 2020 Predictive Maintenance Dataset could not")
    print("  be downloaded automatically. Please follow these steps:")
    print()
    print("  1. Open your browser and go to:")
    print("     https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset")
    print()
    print("  2. Click the 'Download' button to download the ZIP archive.")
    print()
    print("  3. Extract the ZIP file.")
    print()
    print("  4. Locate the file named 'ai4i2020.csv' (or similar).")
    print()
    print("  5. Copy or move it to:")
    print(f"     {(RAW_DIR / TARGET_FILENAME).resolve()}")
    print()
    print("  6. Then re-run the training pipeline:")
    print("     python scripts/run_pipeline.py")
    print()
    print("  DATASET CITATION:")
    print("  Matzka, S. (2020). AI4I 2020 Predictive Maintenance Dataset.")
    print("  UCI Machine Learning Repository.")
    print("  https://doi.org/10.24432/C5HS5C")
    print()
    print("  NOTE: This dataset is SYNTHETIC — generated to model realistic")
    print("  industrial conditions, not from real machine telemetry.")
    print("═" * 65 + "\n")


if __name__ == "__main__":
    ensure_dirs(RAW_DIR)

    # Check if dataset already exists
    existing = list(RAW_DIR.glob("*.csv"))
    if existing:
        logger.info("Dataset already present: %s", existing[0])
        sys.exit(0)

    # Try ucimlrepo package first
    success = try_ucimlrepo()

    # Fall back to direct URL download
    if not success:
        success = download_from_uci()

    if not success:
        print_manual_instructions()
        sys.exit(1)
    else:
        print("\n[OK] Dataset ready. You can now run: python scripts/run_pipeline.py\n")
        sys.exit(0)
