"""Setup script for the Industrial Surface Defect Detection Platform."""

from setuptools import setup, find_packages
from pathlib import Path


def read_requirements():
    """Read dependencies from requirements.txt, ignoring comments and empty lines."""
    requirements_path = Path(__file__).parent / "requirements.txt"
    requirements = []
    with open(requirements_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                requirements.append(line)
    return requirements


setup(
    name="defect-detection",
    version="1.0.0",
    description="Industrial Surface Defect Detection Platform using NEU Surface Defect Dataset",
    author="Defect Detection Team",
    author_email="team@defect-detection.dev",
    python_requires=">=3.10",
    packages=find_packages(include=["src", "src.*", "api", "api.*", "frontend", "frontend.*", "config", "config.*", "mlops", "mlops.*"]),
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "defect-train=scripts.train:main",
            "defect-predict=scripts.export_model:main",
            "defect-evaluate=scripts.evaluate:main",
        ],
    },
)
