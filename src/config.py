"""
Configuration module for DGCNN Mid-Level Fusion model.
Centralizes all hyperparameters and paths.
"""

import os
import torch
from pathlib import Path


class Config:
    """Main configuration class for the project."""
    
    # ============================================================
    # PATHS
    # ============================================================
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
    PLOT_DIR = PROJECT_ROOT / "outputs" / "plots"
    RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
    LOGS_DIR = PROJECT_ROOT / "outputs" / "logs"
    
    # Dataset path (modify according to your setup)
    DATA_PATH = DATA_DIR / "fractal_data_v2_filtered.npy"
    
    # ============================================================
    # MODEL HYPERPARAMETERS
    # ============================================================
    K = 20                    # Number of neighbors in KNN graph
    DROPOUT = 0.5            # Dropout rate
    NUM_CLASSES = 6          # Number of output classes
    N_POINTS = 2048          # Number of points per sample
    
    # ============================================================
    # TRAINING HYPERPARAMETERS
    # ============================================================
    BATCH_SIZE = 32
    EPOCHS = 100
    LR = 1e-3                # Learning rate
    WEIGHT_DECAY = 1e-4      # L2 regularization
    
    # ============================================================
    # DEVICE & PRECISION
    # ============================================================
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    USE_AMP = True           # Use Automatic Mixed Precision
    NUM_WORKERS = 4          # DataLoader workers
    PIN_MEMORY = True        # Pin memory for faster transfer to GPU
    PREFETCH_FACTOR = 2      # Prefetch factor for DataLoader
    
    # ============================================================
    # CLASS CONFIGURATION
    # ============================================================
    CLASS_NAMES = [
        "Other", "Ground", "Low Vegetation",
        "Medium Vegetation", "High Vegetation", "Building"
    ]
    
    # Class remapping from original labels
    FRACTAL_REMAP = {
        1:  0,   # Other
        2:  1,   # Ground
        3:  2,   # Low Vegetation
        4:  3,   # Medium Vegetation
        5:  4,   # High Vegetation
        6:  5,   # Building
        9:  0, 17: 0, 64: 0, 65: 0, 66: 0,  # Map noise to Other
    }
    
    # Class weights for weighted loss (inversely proportional to class frequency)
    CLASS_WEIGHTS = torch.tensor([
        3.0, 0.7, 6.0, 5.0, 0.5, 3.5
    ], dtype=torch.float32)
    
    # ============================================================
    # DATA SPLIT RATIOS
    # ============================================================
    TRAIN_SPLIT = 0.8       # 80% training
    VAL_SPLIT = 0.1         # 10% validation
    TEST_SPLIT = 0.1        # 10% test
    
    # ============================================================
    # AUGMENTATION
    # ============================================================
    AUGMENT_TRAIN = True     # Apply augmentation to training data
    AUGMENT_VAL = False      # No augmentation for validation
    AUGMENT_TEST = False     # No augmentation for test
    
    # Augmentation parameters
    ROTATION_NOISE = 0.01    # Std dev of Gaussian noise for rotation
    SPECTRAL_AUG_RANGE = (0.9, 1.1)  # Spectral channel scaling range
    
    def __init__(self):
        """Initialize config and create necessary directories."""
        self._create_directories()
        self._validate_config()
    
    def _create_directories(self):
        """Create necessary output directories."""
        for directory in [self.OUTPUT_DIR, self.PLOT_DIR, 
                         self.RESULTS_DIR, self.LOGS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _validate_config(self):
        """Validate configuration values."""
        assert self.NUM_CLASSES == len(self.CLASS_NAMES), \
            "Mismatch between NUM_CLASSES and CLASS_NAMES"
        assert self.NUM_CLASSES == len(self.CLASS_WEIGHTS), \
            "Mismatch between NUM_CLASSES and CLASS_WEIGHTS"
        assert self.TRAIN_SPLIT + self.VAL_SPLIT + self.TEST_SPLIT == 1.0, \
            "Data split ratios must sum to 1.0"
    
    def print_summary(self):
        """Print configuration summary."""
        print("=" * 70)
        print("CONFIGURATION SUMMARY")
        print("=" * 70)
        print(f"Device             : {self.DEVICE}")
        print(f"Batch Size         : {self.BATCH_SIZE}")
        print(f"Epochs             : {self.EPOCHS}")
        print(f"Learning Rate      : {self.LR}")
        print(f"Number of Points   : {self.N_POINTS}")
        print(f"K (KNN neighbors)  : {self.K}")
        print(f"Dropout            : {self.DROPOUT}")
        print(f"Use AMP            : {self.USE_AMP}")
        print(f"Output Directory   : {self.OUTPUT_DIR}")
        print("=" * 70)
    
    def to_dict(self):
        """Convert config to dictionary."""
        return {
            'batch_size': self.BATCH_SIZE,
            'epochs': self.EPOCHS,
            'lr': self.LR,
            'weight_decay': self.WEIGHT_DECAY,
            'n_points': self.N_POINTS,
            'k': self.K,
            'dropout': self.DROPOUT,
            'num_classes': self.NUM_CLASSES,
        }


# Create a global config instance
config = Config()
