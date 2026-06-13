"""
Global configuration for the Industrial Surface Defect Detection Platform.

This module provides centralized configuration for paths, model parameters,
dataset information, class names, and environment-specific settings.

Usage:
    from config.settings import *
    or
    from config.settings import CLASS_NAMES, DATA_DIR, DEVICE
"""

import os
from pathlib import Path

import torch
from dotenv import load_dotenv

# =============================================================================
# Environment Setup
# =============================================================================

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# Project Paths
# =============================================================================

# Project root is the parent of the config/ directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
YOLO_DATA_DIR = DATA_DIR / "yolo"

# Model and output directories
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"

# =============================================================================
# Dataset Configuration
# =============================================================================

CLASS_NAMES = [
    "Crazing",
    "Inclusion",
    "Patches",
    "Pitted_Surface",
    "Rolled-in_Scale",
    "Scratches",
]

NUM_CLASSES = 6

# Image dimensions
IMAGE_SIZE_CLASSIFICATION = (224, 224)
IMAGE_SIZE_DETECTION = (640, 640)
ORIGINAL_IMAGE_SIZE = (200, 200)

# =============================================================================
# Training Defaults
# =============================================================================

BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001

# Dataset split ratios
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Reproducibility
RANDOM_SEED = 42

# =============================================================================
# Normalization (ImageNet Statistics)
# =============================================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# =============================================================================
# Inference Defaults
# =============================================================================

CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# Database Configuration
# =============================================================================

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/defect_detection",
)

# =============================================================================
# API Configuration
# =============================================================================

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
