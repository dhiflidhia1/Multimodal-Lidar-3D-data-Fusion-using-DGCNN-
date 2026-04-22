"""
Data loading and preprocessing module.
Handles dataset creation, augmentation, and DataLoader setup.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


class FRACTALDataset(Dataset):
    """
    FRACTAL point cloud dataset with multi-modal data.
    
    Features:
    - Points: X, Y, Z, ReturnNum, NumReturns (spatial)
    - Colors: R, G, B, IR, NDVI (spectral)
    - Labels: semantic class label
    """
    
    def __init__(self, data: np.ndarray, augment: bool = False, 
                 n_points: int = 2048, class_remap: dict = None):
        """
        Initialize dataset.
        
        Args:
            data: numpy array of shape (N_samples, N_points, N_features)
                  Last column is the label
            augment: whether to apply data augmentation
            n_points: number of points to sample per cloud
            class_remap: dictionary mapping original labels to new labels
        """
        self.augment = augment
        self.n_points = n_points
        self.class_remap = class_remap or {}
        self.samples = []
        
        # Parse data
        for i in range(len(data)):
            self.samples.append({
                "pts": data[i, :, :-1].astype(np.float32),
                "labels": data[i, :, -1].astype(np.int64),
            })
        
        logger.info(f"Loaded {len(self.samples)} samples")
    
    def _remap_labels(self, labels: np.ndarray) -> np.ndarray:
        """Remap original labels to training labels."""
        remapped = np.zeros_like(labels)
        for orig_label, new_label in self.class_remap.items():
            remapped[labels == orig_label] = new_label
        return remapped
    
    def _compute_ndvi(self, red: np.ndarray, ir: np.ndarray) -> np.ndarray:
        """
        Compute Normalized Difference Vegetation Index (NDVI).
        NDVI = (IR - R) / (IR + R + eps)
        """
        ndvi = np.clip((ir - red) / (ir + red + 1e-8), -1.0, 1.0)
        return ndvi
    
    def _augment_spatial(self, spatial: np.ndarray) -> np.ndarray:
        """Apply random rotation and noise to spatial coordinates."""
        # Random rotation around Z-axis
        theta = np.random.uniform(0, 2 * np.pi)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rot_matrix = np.array([
            [cos_t, -sin_t, 0],
            [sin_t, cos_t, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        
        spatial[:, :3] = spatial[:, :3] @ rot_matrix.T
        
        # Add Gaussian noise
        spatial[:, :3] += np.random.normal(0, 0.01, (len(spatial), 3)).astype(np.float32)
        
        return spatial
    
    def _augment_spectral(self, spectral: np.ndarray) -> np.ndarray:
        """Apply random scaling to spectral channels."""
        if np.random.random() > 0.5:
            scale = np.random.uniform(0.9, 1.1)
            spectral[:, :4] *= scale
            spectral[:, :4] = np.clip(spectral[:, :4], 0, 1)
        return spectral
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a sample.
        
        Returns:
            spatial: (N_points, 5) - X, Y, Z, ReturnNum, NumReturns
            spectral: (N_points, 5) - R, G, B, IR, NDVI
            labels: (N_points,) - semantic labels
        """
        sample = self.samples[idx]
        pts = sample["pts"].copy()
        labels = sample["labels"].copy()
        
        # Remap labels
        labels = self._remap_labels(labels)
        
        # Subsample if necessary
        n_available = len(pts)
        if n_available > self.n_points:
            choice = np.random.choice(n_available, self.n_points, replace=False)
            pts = pts[choice]
            labels = labels[choice]
        elif n_available < self.n_points:
            # Padding: repeat random points
            choice = np.random.choice(n_available, self.n_points - n_available, replace=True)
            pts = np.vstack([pts, pts[choice]])
            labels = np.concatenate([labels, labels[choice]])
        
        # Extract features
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        r, g, b = pts[:, 3], pts[:, 4], pts[:, 5]
        ir = pts[:, 6]
        return_num = pts[:, 7]
        num_returns = pts[:, 8]
        
        # Compute NDVI
        ndvi = self._compute_ndvi(r, ir)
        
        # Build spatial and spectral stacks
        spatial = np.stack([x, y, z, return_num, num_returns], axis=1)
        spectral = np.stack([r, g, b, ir, ndvi], axis=1)
        
        # Normalize
        spatial = (spatial - spatial.mean(0)) / (spatial.std(0) + 1e-6)
        spectral = (spectral - spectral.mean(0)) / (spectral.std(0) + 1e-6)
        
        # Augmentation
        if self.augment:
            spatial = self._augment_spatial(spatial)
            spectral = self._augment_spectral(spectral)
        
        return (
            torch.from_numpy(spatial.astype(np.float32)),
            torch.from_numpy(spectral.astype(np.float32)),
            torch.from_numpy(labels)
        )


def load_data(data_path: str, config) -> Tuple[np.ndarray, int, int, int]:
    """
    Load data and compute split sizes.
    
    Args:
        data_path: path to .npy file
        config: Config object
    
    Returns:
        all_data: full dataset
        n_train: number of training samples
        n_val: number of validation samples
    """
    logger.info(f"Loading data from {data_path}")
    all_data = np.load(data_path)
    logger.info(f"Data shape: {all_data.shape}")
    
    n_total = len(all_data)
    n_train = int(n_total * config.TRAIN_SPLIT)
    n_val = int(n_total * config.VAL_SPLIT)
    n_test = n_total - n_train - n_val
    
    logger.info(f"Split: train={n_train}, val={n_val}, test={n_test}")
    
    return all_data, n_train, n_val, n_test


def create_dataloaders(data_path: str, config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, val, and test dataloaders.
    
    Args:
        data_path: path to .npy file
        config: Config object
    
    Returns:
        train_loader, val_loader, test_loader
    """
    all_data, n_train, n_val, n_test = load_data(data_path, config)
    
    # Create datasets
    logger.info("Creating training dataset")
    train_ds = FRACTALDataset(
        all_data[:n_train],
        augment=config.AUGMENT_TRAIN,
        n_points=config.N_POINTS,
        class_remap=config.FRACTAL_REMAP
    )
    
    logger.info("Creating validation dataset")
    val_ds = FRACTALDataset(
        all_data[n_train:n_train + n_val],
        augment=config.AUGMENT_VAL,
        n_points=config.N_POINTS,
        class_remap=config.FRACTAL_REMAP
    )
    
    logger.info("Creating test dataset")
    test_ds = FRACTALDataset(
        all_data[n_train + n_val:],
        augment=config.AUGMENT_TEST,
        n_points=config.N_POINTS,
        class_remap=config.FRACTAL_REMAP
    )
    
    del all_data  # Free memory
    
    # Create dataloaders
    def get_loader(dataset, shuffle=False):
        return DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=shuffle,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            prefetch_factor=config.PREFETCH_FACTOR,
            persistent_workers=config.NUM_WORKERS > 0
        )
    
    train_loader = get_loader(train_ds, shuffle=True)
    val_loader = get_loader(val_ds, shuffle=False)
    test_loader = get_loader(test_ds, shuffle=False)
    
    logger.info(f"Created dataloaders: "
                f"train={len(train_loader)}, "
                f"val={len(val_loader)}, "
                f"test={len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader
