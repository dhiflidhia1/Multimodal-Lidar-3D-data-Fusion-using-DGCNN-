"""
DGCNN Mid-Level Fusion for Semantic Segmentation on Point Clouds.

Main package providing modular components for training and evaluation.
"""

from .config import Config, config
from .data import FRACTALDataset, load_data, create_dataloaders
from .model import DGCNN_MidFusion, DGCNNSpatialBranch, SpectralBranch, create_model
from .metrics import compute_metrics, MetricAggregator, print_metrics
from .training import Trainer, create_optimizer, create_criterion
from .visualization import ResultsVisualizer

__version__ = "1.0.0"
__all__ = [
    'Config', 'config',
    'FRACTALDataset', 'load_data', 'create_dataloaders',
    'DGCNN_MidFusion', 'DGCNNSpatialBranch', 'SpectralBranch', 'create_model',
    'compute_metrics', 'MetricAggregator', 'print_metrics',
    'Trainer', 'create_optimizer', 'create_criterion',
    'ResultsVisualizer',
]
