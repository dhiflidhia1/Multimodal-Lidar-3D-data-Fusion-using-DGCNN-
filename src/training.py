"""
Training and evaluation loops.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import logging
from typing import Tuple, Dict

from .metrics import MetricAggregator, to_numpy, compute_metrics

logger = logging.getLogger(__name__)


class Trainer:
    """Handles training and evaluation loops."""
    
    def __init__(self, model: nn.Module, criterion: nn.Module, 
                 optimizer, config, device: torch.device):
        """
        Initialize trainer.
        
        Args:
            model: neural network model
            criterion: loss function
            optimizer: optimizer
            config: Config object
            device: torch device
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.device = device
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            optimizer, T_max=config.EPOCHS, eta_min=1e-5
        )
        
        # Mixed precision training
        self.scaler = GradScaler('cuda') if config.USE_AMP else None
        self.use_amp = config.USE_AMP
    
    def train_one_epoch(self, train_loader: DataLoader) -> Tuple[float, Dict]:
        """
        Train for one epoch.
        
        Args:
            train_loader: training dataloader
        
        Returns:
            avg_loss: average loss over epoch
            metrics: evaluation metrics
        """
        self.model.train()
        aggregator = MetricAggregator()
        
        pbar = tqdm(train_loader, desc="Train", leave=False)
        for spatial, spectral, labels in pbar:
            spatial = spatial.to(self.device, non_blocking=True)
            spectral = spectral.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Forward pass
            self.optimizer.zero_grad(set_to_none=True)
            
            with autocast('cuda', enabled=self.use_amp):
                logits = self.model(spatial, spectral)  # (B, C, N)
                loss = self.criterion(logits, labels)
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            # Update metrics
            preds = logits.argmax(dim=1).cpu().numpy().reshape(-1)
            labels_np = labels.cpu().numpy().reshape(-1)
            aggregator.update(preds, labels_np, loss.item())
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Compute metrics
        all_preds, all_labels, avg_loss = aggregator.get_aggregated()
        metrics = compute_metrics(
            all_preds, all_labels, self.config.NUM_CLASSES, 
            self.config.CLASS_NAMES
        )
        
        return avg_loss, metrics
    
    def evaluate(self, val_loader: DataLoader) -> Tuple[float, Dict, np.ndarray, np.ndarray]:
        """
        Evaluate on validation/test set.
        
        Args:
            val_loader: validation/test dataloader
        
        Returns:
            avg_loss: average loss
            metrics: evaluation metrics
            predictions: all predictions
            labels: all labels
        """
        self.model.eval()
        aggregator = MetricAggregator()
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Eval", leave=False)
            for spatial, spectral, labels in pbar:
                spatial = spatial.to(self.device, non_blocking=True)
                spectral = spectral.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                with autocast('cuda', enabled=self.use_amp):
                    logits = self.model(spatial, spectral)
                    loss = self.criterion(logits, labels)
                
                preds = logits.argmax(dim=1).cpu().numpy().reshape(-1)
                labels_np = labels.cpu().numpy().reshape(-1)
                aggregator.update(preds, labels_np, loss.item())
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        all_preds, all_labels, avg_loss = aggregator.get_aggregated()
        metrics = compute_metrics(
            all_preds, all_labels, self.config.NUM_CLASSES,
            self.config.CLASS_NAMES
        )
        
        return avg_loss, metrics, all_preds, all_labels
    
    def step_scheduler(self):
        """Step learning rate scheduler."""
        self.scheduler.step()
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']


def format_time(seconds: float) -> str:
    """Format seconds to readable time string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.2f}h"


def print_epoch_summary(epoch: int, total_epochs: int, train_loss: float, 
                       val_loss: float, metrics: Dict, 
                       elapsed_time: float, remaining_time: float = None):
    """Pretty-print epoch training summary."""
    print(f"\nEpoch {epoch:3d}/{total_epochs} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"OA: {metrics['OA']*100:.1f}% | "
          f"mIoU: {metrics['mIoU']*100:.1f}% | "
          f"F1: {metrics['macro_f1']*100:.1f}%")
    
    print(f"Time: {format_time(elapsed_time)}", end="")
    if remaining_time is not None:
        print(f" | Remaining: {format_time(remaining_time)}")
    else:
        print()


def create_optimizer(model: nn.Module, config):
    """
    Create Adam optimizer.
    
    Args:
        model: torch model
        config: Config object
    
    Returns:
        optimizer
    """
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY
    )
    return optimizer


def create_criterion(config, device):
    """
    Create weighted cross-entropy loss.
    
    Args:
        config: Config object
        device: torch device
    
    Returns:
        loss function
    """
    weights = config.CLASS_WEIGHTS.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    return criterion
