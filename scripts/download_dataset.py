"""
Download the NEU Surface Defect Dataset.

This script downloads the NEU Surface Defect Dataset (1,800 grayscale images,
200x200px, 6 classes × 300 images each) and organizes it into class subfolders
under data/raw/images/.

Dataset classes:
    - Crazing (300 images)
    - Inclusion (300 images)
    - Patches (300 images)
    - Pitted_Surface (300 images)
    - Rolled-in_Scale (300 images)
    - Scratches (300 images)

Methods:
    1. KaggleHub download (simplest, pip install kagglehub)
    2. Kaggle CLI download (requires authentication)
    3. Manual download instructions as fallback

Usage:
    python scripts/download_dataset.py
    python scripts/download_dataset.py --output-dir /path/to/custom/dir
"""

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

# Add project root to path so we can import config and src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import CLASS_NAMES, RAW_DATA_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

# =============================================================================
# Constants
# =============================================================================

KAGGLE_DATASET = "kaustubhdikshit/neu-surface-defect-database"
EXPECTED_TOTAL_IMAGES = 1800
EXPECTED_IMAGES_PER_CLASS = 300
IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


# =============================================================================
# Verification
# =============================================================================


def verify_dataset(images_dir: Path) -> bool:
    """
    Verify that the dataset is correctly downloaded and organized.

    Checks that each class subfolder exists and contains the expected number
    of images (300 per class, 1800 total).

    Args:
        images_dir: Path to data/raw/images/ directory.

    Returns:
        True if dataset is complete and valid, False otherwise.
    """
    if not images_dir.exists():
        return False

    total_count = 0
    for class_name in CLASS_NAMES:
        class_dir = images_dir / class_name
        if not class_dir.exists():
            logger.warning(f"Missing class directory: {class_name}")
            return False

        image_files = [
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ]
        count = len(image_files)
        total_count += count

        if count != EXPECTED_IMAGES_PER_CLASS:
            logger.warning(
                f"Class '{class_name}' has {count} images, expected {EXPECTED_IMAGES_PER_CLASS}"
            )
            return False

    if total_count != EXPECTED_TOTAL_IMAGES:
        logger.warning(
            f"Total image count is {total_count}, expected {EXPECTED_TOTAL_IMAGES}"
        )
        return False

    return True


def dataset_exists(images_dir: Path) -> bool:
    """
    Check if the dataset already exists and is complete.

    Args:
        images_dir: Path to the images directory.

    Returns:
        True if dataset already downloaded and valid.
    """
    if not images_dir.exists():
        return False

    # Quick check: do all class directories exist with files?
    for class_name in CLASS_NAMES:
        class_dir = images_dir / class_name
        if not class_dir.exists():
            return False
        # Check if there are at least some image files
        image_files = [
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if len(image_files) == 0:
            return False

    return True


# =============================================================================
# Download Methods
# =============================================================================


def download_with_kagglehub(output_dir: Path) -> bool:
    """
    Download the NEU dataset using kagglehub (simplest method).

    Requires:
        - kagglehub package installed (pip install kagglehub)

    Args:
        output_dir: Directory where the dataset will be organized.

    Returns:
        True if download was successful, False otherwise.
    """
    logger.info("Attempting download via kagglehub...")

    try:
        import kagglehub
    except ImportError:
        logger.warning(
            "kagglehub not installed. Install with: pip install kagglehub"
        )
        return False

    try:
        # Download latest version
        path = kagglehub.dataset_download("kaustubhdikshit/neu-surface-defect-database")
        logger.info(f"KaggleHub download complete. Path: {path}")
        print(f"Path to dataset files: {path}")

        # Organize the downloaded files into class subfolders
        source_dir = Path(path)
        return _organize_downloaded_files(source_dir, output_dir)

    except Exception as e:
        logger.error(f"KaggleHub download failed: {e}")
        return False


def download_with_kaggle_cli(output_dir: Path) -> bool:
    """
    Download the NEU dataset using the Kaggle CLI.

    Requires:
        - kaggle package installed (pip install kaggle)
        - Kaggle API credentials at ~/.kaggle/kaggle.json or via
          KAGGLE_USERNAME and KAGGLE_KEY environment variables.

    Args:
        output_dir: Directory where the dataset will be extracted.

    Returns:
        True if download was successful, False otherwise.
    """
    logger.info("Attempting download via Kaggle CLI...")

    # Check if kaggle CLI is available
    try:
        result = subprocess.run(
            ["kaggle", "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            logger.warning("Kaggle CLI is installed but returned an error.")
            return False
    except FileNotFoundError:
        logger.warning(
            "Kaggle CLI not found. Install it with: pip install kaggle"
        )
        return False
    except subprocess.TimeoutExpired:
        logger.warning("Kaggle CLI version check timed out.")
        return False

    # Check for authentication
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    has_env_creds = (
        os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")
    )

    if not kaggle_json.exists() and not has_env_creds:
        logger.warning(
            "Kaggle credentials not found. Either place kaggle.json in "
            "~/.kaggle/ or set KAGGLE_USERNAME and KAGGLE_KEY environment variables."
        )
        return False

    # Create a temp download directory
    temp_dir = output_dir / "_kaggle_download"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"Downloading dataset: {KAGGLE_DATASET}")
        result = subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", KAGGLE_DATASET,
                "-p", str(temp_dir),
                "--unzip",
            ],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout for download
        )

        if result.returncode != 0:
            logger.error(f"Kaggle download failed: {result.stderr}")
            return False

        logger.info("Kaggle download complete. Organizing files...")
        return _organize_downloaded_files(temp_dir, output_dir)

    except subprocess.TimeoutExpired:
        logger.error("Kaggle download timed out after 10 minutes.")
        return False
    except Exception as e:
        logger.error(f"Kaggle download error: {e}")
        return False
    finally:
        # Clean up temp directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def download_with_kaggle_api(output_dir: Path) -> bool:
    """
    Download the NEU dataset using the Kaggle Python API directly.

    This method uses the kaggle Python package programmatically rather than
    the CLI. Requires the same authentication setup as the CLI method.

    Args:
        output_dir: Directory where the dataset will be extracted.

    Returns:
        True if download was successful, False otherwise.
    """
    logger.info("Attempting download via Kaggle Python API...")

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        logger.warning(
            "Kaggle Python package not installed. Install with: pip install kaggle"
        )
        return False

    try:
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        logger.warning(f"Kaggle authentication failed: {e}")
        return False

    temp_dir = output_dir / "_kaggle_api_download"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"Downloading dataset: {KAGGLE_DATASET}")
        api.dataset_download_files(
            KAGGLE_DATASET,
            path=str(temp_dir),
            unzip=True,
        )

        logger.info("Download complete. Organizing files...")
        return _organize_downloaded_files(temp_dir, output_dir)

    except Exception as e:
        logger.error(f"Kaggle API download error: {e}")
        return False
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def _organize_downloaded_files(source_dir: Path, images_dir: Path) -> bool:
    """
    Organize downloaded files into class subfolders.

    The NEU dataset images are named like: Crazing_1.bmp, Inclusion_1.bmp, etc.
    This function places them into their respective class directories.

    Args:
        source_dir: Directory containing the downloaded/extracted files.
        images_dir: Target directory (data/raw/images/) for organized output.

    Returns:
        True if organization was successful, False otherwise.
    """
    images_dir.mkdir(parents=True, exist_ok=True)

    # Create class directories
    for class_name in CLASS_NAMES:
        (images_dir / class_name).mkdir(parents=True, exist_ok=True)

    # Find all image files recursively in source
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(source_dir.rglob(f"*{ext}"))
        image_files.extend(source_dir.rglob(f"*{ext.upper()}"))

    # Remove duplicates (rglob may find same file via different patterns)
    image_files = list(set(image_files))

    if not image_files:
        logger.error("No image files found in downloaded data.")
        return False

    logger.info(f"Found {len(image_files)} image files. Organizing by class...")

    # Check if files are already in class subfolders
    organized_count = 0
    for img_path in image_files:
        # Determine class from filename or parent directory
        class_name = _determine_class(img_path)
        if class_name is None:
            logger.debug(f"Could not determine class for: {img_path.name}")
            continue

        target_dir = images_dir / class_name
        target_path = target_dir / img_path.name

        if not target_path.exists():
            shutil.copy2(str(img_path), str(target_path))
        organized_count += 1

    logger.info(f"Organized {organized_count} images into class folders.")
    return organized_count > 0


def _determine_class(img_path: Path) -> str | None:
    """
    Determine the class of an image from its filename or parent folder.

    NEU images are named like: Crazing_1.bmp, Inclusion_42.bmp, etc.
    They may also already be in class-named subfolders.

    Args:
        img_path: Path to the image file.

    Returns:
        The class name string, or None if it cannot be determined.
    """
    filename = img_path.stem  # e.g., "Crazing_1" or "Rolled-in_Scale_123"

    # First, check if the parent directory matches a class name
    parent_name = img_path.parent.name
    if parent_name in CLASS_NAMES:
        return parent_name

    # Try to match filename prefix to class names
    # Sort by length (descending) to match longest prefix first
    # e.g., "Rolled-in_Scale" before "Rolled"
    sorted_classes = sorted(CLASS_NAMES, key=len, reverse=True)
    for class_name in sorted_classes:
        if filename.startswith(class_name):
            return class_name

    # Handle case-insensitive matching
    filename_lower = filename.lower()
    for class_name in sorted_classes:
        if filename_lower.startswith(class_name.lower()):
            return class_name

    return None


def extract_zip_with_progress(zip_path: Path, extract_to: Path) -> None:
    """
    Extract a zip file with progress reporting.

    Args:
        zip_path: Path to the zip file.
        extract_to: Directory to extract contents into.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    with zipfile.ZipFile(str(zip_path), "r") as zf:
        members = zf.namelist()
        if tqdm:
            for member in tqdm(members, desc="Extracting", unit="file"):
                zf.extract(member, str(extract_to))
        else:
            logger.info(f"Extracting {len(members)} files...")
            zf.extractall(str(extract_to))


# =============================================================================
# Manual Download Fallback
# =============================================================================


def print_manual_instructions(images_dir: Path) -> None:
    """
    Print instructions for manually downloading the dataset.

    Args:
        images_dir: The target directory where images should be placed.
    """
    instructions = f"""
╔══════════════════════════════════════════════════════════════════════╗
║              Manual Download Instructions                            ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  The automatic download methods were not available.                   ║
║  Please download the dataset manually:                               ║
║                                                                      ║
║  1. Go to: https://www.kaggle.com/datasets/kaustubhdikshit/         ║
║            neu-surface-defect-database                                ║
║                                                                      ║
║  2. Click "Download" (requires Kaggle account)                       ║
║                                                                      ║
║  3. Extract the ZIP file                                             ║
║                                                                      ║
║  4. Organize images into class subfolders at:                        ║
║     {str(images_dir):<55}║
║                                                                      ║
║  Expected structure:                                                 ║
║     images/                                                          ║
║     ├── Crazing/         (300 .bmp images)                           ║
║     ├── Inclusion/       (300 .bmp images)                           ║
║     ├── Patches/         (300 .bmp images)                           ║
║     ├── Pitted_Surface/  (300 .bmp images)                           ║
║     ├── Rolled-in_Scale/ (300 .bmp images)                           ║
║     └── Scratches/       (300 .bmp images)                           ║
║                                                                      ║
║  Alternative: Set up Kaggle CLI                                      ║
║     pip install kaggle                                               ║
║     # Place kaggle.json in ~/.kaggle/                                ║
║     # Or set KAGGLE_USERNAME and KAGGLE_KEY env vars                 ║
║     python scripts/download_dataset.py                               ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""
    print(instructions)
    logger.info(
        "Manual download instructions displayed. "
        f"Target directory: {images_dir}"
    )


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """
    Main entry point for downloading the NEU Surface Defect Dataset.

    Attempts download in order:
        1. Kaggle CLI (if available and authenticated)
        2. Kaggle Python API (if available and authenticated)
        3. Falls back to manual download instructions

    Skips download if dataset already exists and is verified.
    """
    parser = argparse.ArgumentParser(
        description="Download the NEU Surface Defect Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/download_dataset.py
    python scripts/download_dataset.py --output-dir ./custom_data/raw/images
    python scripts/download_dataset.py --verify-only
        """,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Custom output directory for images (default: data/raw/images/)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing dataset without downloading",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if dataset exists",
    )

    args = parser.parse_args()

    # Determine output directory
    images_dir = args.output_dir if args.output_dir else RAW_DATA_DIR / "images"
    images_dir = images_dir.resolve()

    logger.info(f"Target directory: {images_dir}")
    logger.info(f"Expected: {EXPECTED_TOTAL_IMAGES} images across {len(CLASS_NAMES)} classes")

    # Verify-only mode
    if args.verify_only:
        if verify_dataset(images_dir):
            logger.info("✓ Dataset verification PASSED. All images present and accounted for.")
            print(f"\n✓ Dataset is complete: {EXPECTED_TOTAL_IMAGES} images in {len(CLASS_NAMES)} classes.")
        else:
            logger.error("✗ Dataset verification FAILED. Dataset is incomplete or missing.")
            print("\n✗ Dataset is incomplete or missing.")
            sys.exit(1)
        return

    # Check if dataset already exists
    if not args.force and dataset_exists(images_dir):
        if verify_dataset(images_dir):
            logger.info("Dataset already exists and is verified. Skipping download.")
            print(f"\n✓ Dataset already exists at: {images_dir}")
            print(f"  {EXPECTED_TOTAL_IMAGES} images across {len(CLASS_NAMES)} classes.")
            print("  Use --force to re-download.")
            return
        else:
            logger.warning(
                "Dataset directory exists but verification failed. Re-downloading..."
            )

    # Ensure parent directory exists
    images_dir.mkdir(parents=True, exist_ok=True)

    # Attempt download methods in order
    download_success = False

    # Method 1: KaggleHub (simplest - pip install kagglehub)
    if not download_success:
        download_success = download_with_kagglehub(images_dir)

    # Method 2: Kaggle CLI
    if not download_success:
        download_success = download_with_kaggle_cli(images_dir)

    # Method 3: Kaggle Python API
    if not download_success:
        download_success = download_with_kaggle_api(images_dir)

    # If download succeeded, verify
    if download_success:
        logger.info("Download complete. Verifying dataset...")
        if verify_dataset(images_dir):
            logger.info("✓ Dataset download and verification successful!")
            print(f"\n✓ Successfully downloaded NEU Surface Defect Dataset to: {images_dir}")
            print(f"  Total images: {EXPECTED_TOTAL_IMAGES}")
            print(f"  Classes: {', '.join(CLASS_NAMES)}")
            print(f"  Images per class: {EXPECTED_IMAGES_PER_CLASS}")
        else:
            logger.warning(
                "Download completed but verification failed. "
                "Some images may be missing. Check the logs for details."
            )
            print("\n⚠ Download completed but dataset may be incomplete.")
            print("  Run with --verify-only to see details.")
    else:
        # Fallback: print manual instructions
        logger.warning("All automatic download methods failed.")
        print_manual_instructions(images_dir)
        sys.exit(1)


if __name__ == "__main__":
    main()
