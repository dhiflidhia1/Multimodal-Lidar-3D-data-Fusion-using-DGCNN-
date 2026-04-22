"""
Metrics computation and utilities for training/evaluation.
"""

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)


def compute_metrics(predictions: np.ndarray, labels: np.ndarray, 
                   num_classes: int, class_names: List[str] = None) -> Dict:
    """
    Compute comprehensive metrics for semantic segmentation.
    
    Args:
        predictions: (N,) predicted class indices
        labels: (N,) ground truth labels
        num_classes: number of classes
        class_names: optional list of class names
    
    Returns:
        Dictionary with metrics:
        - OA: Overall Accuracy
        - mIoU: mean Intersection-over-Union
        - macro_f1: macro F1-score
        - per_class_iou: list of per-class IoU values
        - per_class_f1: list of per-class F1 scores
        - per_class_precision: per-class precision
        - per_class_recall: per-class recall
    """
    
    # Per-class IoU (Intersection over Union)
    ious = []
    for cls_idx in range(num_classes):
        tp = np.sum((predictions == cls_idx) & (labels == cls_idx))
        fp = np.sum((predictions == cls_idx) & (labels != cls_idx))
        fn = np.sum((predictions != cls_idx) & (labels == cls_idx))
        
        denominator = tp + fp + fn
        iou = tp / denominator if denominator > 0 else 0.0
        ious.append(iou)
    
    # Overall Accuracy
    oa = float(np.mean(predictions == labels))
    
    # Mean IoU
    miou = float(np.mean(ious))
    
    # Per-class metrics using sklearn
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, labels=range(num_classes), zero_division=0
    )
    
    metrics = {
        "OA": oa,
        "mIoU": miou,
        "macro_f1": float(f1.mean()),
        "per_class_iou": ious,
        "per_class_f1": f1.tolist(),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
    }
    
    return metrics


def get_confusion_matrix(predictions: np.ndarray, labels: np.ndarray, 
                        num_classes: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute confusion matrix and normalized version.
    
    Args:
        predictions: (N,) predicted labels
        labels: (N,) ground truth labels
        num_classes: number of classes
    
    Returns:
        cm: raw confusion matrix
        cm_norm: normalized confusion matrix (percentages)
    """
    cm = confusion_matrix(labels, predictions, labels=range(num_classes))
    cm_norm = cm.astype(float) / (cm.sum(axis=1)[:, None] + 1e-10) * 100
    return cm, cm_norm


class MetricAggregator:
    """Accumulate predictions and labels for batch-wise evaluation."""
    
    def __init__(self):
        self.predictions = []
        self.labels = []
        self.losses = []
    
    def update(self, preds: np.ndarray, labels: np.ndarray, loss: float = None):
        """
        Update with new batch.
        
        Args:
            preds: (N,) or logits (B, C, N) -> argmax -> (N,)
            labels: (N,)
            loss: optional batch loss
        """
        if preds.ndim > 1:
            preds = np.argmax(preds, axis=1)
        
        self.predictions.append(preds.reshape(-1))
        self.labels.append(labels.reshape(-1))
        
        if loss is not None:
            self.losses.append(loss)
    
    def get_aggregated(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Get aggregated predictions, labels, and avg loss."""
        all_preds = np.concatenate(self.predictions)
        all_labels = np.concatenate(self.labels)
        avg_loss = float(np.mean(self.losses)) if self.losses else 0.0
        return all_preds, all_labels, avg_loss
    
    def reset(self):
        """Reset accumulated values."""
        self.predictions = []
        self.labels = []
        self.losses = []
    
    def __len__(self) -> int:
        return len(self.predictions)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert torch tensor to numpy array."""
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().detach().numpy()
    return tensor


def print_metrics(metrics: Dict, class_names: List[str], 
                 prefix: str = ""):
    """
    Pretty-print evaluation metrics.
    
    Args:
        metrics: metrics dictionary
        class_names: list of class names
        prefix: prefix for print (e.g., "Val", "Test")
    """
    print(f"\n{prefix} Metrics:")
    print(f"  Overall Accuracy (OA): {metrics['OA']*100:.2f}%")
    print(f"  Mean IoU (mIoU):       {metrics['mIoU']*100:.2f}%")
    print(f"  Macro F1-Score:       {metrics['macro_f1']*100:.2f}%")
    
    print(f"\n{'Class':<20} {'IoU':>8} {'F1':>8} {'Precision':>10} {'Recall':>8}")
    print("─" * 65)
    
    for i, name in enumerate(class_names):
        iou = metrics['per_class_iou'][i]
        f1 = metrics['per_class_f1'][i]
        prec = metrics['per_class_precision'][i]
        rec = metrics['per_class_recall'][i]
        print(f"{name:<20} {iou*100:>7.2f}% {f1*100:>7.2f}% "
              f"{prec*100:>9.2f}% {rec*100:>7.2f}%")


# ============================================================
# Gradient Utilities
# ============================================================

def clip_gradients(model: torch.nn.Module, max_norm: float = 1.0) -> float:
    """
    Clip model gradients by global norm.
    
    Args:
        model: torch model
        max_norm: maximum gradient norm
    
    Returns:
        total_norm: total norm before clipping
    """
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    return float(total_norm)


# ============================================================
# Model Utilities
# ============================================================

def save_checkpoint(model: torch.nn.Module, optimizer, epoch: int, 
                   save_path: str, metrics: Dict = None):
    """
    Save model checkpoint.
    
    Args:
        model: torch model
        optimizer: optimizer
        epoch: epoch number
        save_path: path to save checkpoint
        metrics: optional metrics to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics or {},
    }
    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved: {save_path}")


def load_checkpoint(model: torch.nn.Module, optimizer, load_path: str, device):
    """
    Load model checkpoint.
    
    Args:
        model: torch model
        optimizer: optimizer
        load_path: path to checkpoint
        device: device to load to
    
    Returns:
        epoch: epoch from checkpoint
        metrics: metrics from checkpoint
    """
    checkpoint = torch.load(load_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', {})
    logger.info(f"Checkpoint loaded: {load_path}")
    return epoch, metrics
