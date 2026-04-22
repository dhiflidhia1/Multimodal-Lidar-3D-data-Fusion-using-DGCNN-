"""
Visualization and results saving utilities.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import logging

from .metrics import get_confusion_matrix

logger = logging.getLogger(__name__)


class ResultsVisualizer:
    """Generate and save visualizations."""
    
    def __init__(self, config):
        """
        Initialize visualizer.
        
        Args:
            config: Config object with output directories
        """
        self.config = config
        self.plot_dir = config.PLOT_DIR
        self.results_dir = config.RESULTS_DIR
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
    
    def plot_confusion_matrix(self, predictions: np.ndarray, 
                             labels: np.ndarray, save_name: str = "confusion_matrix.png"):
        """
        Plot and save confusion matrix.
        
        Args:
            predictions: (N,) predicted labels
            labels: (N,) ground truth labels
            save_name: output filename
        """
        cm, cm_norm = get_confusion_matrix(
            predictions, labels, self.config.NUM_CLASSES
        )
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Raw confusion matrix
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=self.config.CLASS_NAMES,
            yticklabels=self.config.CLASS_NAMES,
            cbar_kws={'label': 'Count'}
        )
        axes[0].set_title("Confusion Matrix (Raw Counts)")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Ground Truth")
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].tick_params(axis='y', rotation=0)
        
        # Normalized confusion matrix
        sns.heatmap(
            cm_norm, annot=True, fmt='.1f', cmap='Blues', ax=axes[1],
            xticklabels=self.config.CLASS_NAMES,
            yticklabels=self.config.CLASS_NAMES,
            cbar_kws={'label': 'Percentage (%)'}
        )
        axes[1].set_title("Confusion Matrix (Normalized %)")
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("Ground Truth")
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].tick_params(axis='y', rotation=0)
        
        plt.tight_layout()
        save_path = self.plot_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved confusion matrix: {save_path}")
    
    def plot_training_curves(self, history: Dict, save_name: str = "training_curves.png"):
        """
        Plot training history (loss, mIoU, F1).
        
        Args:
            history: dictionary with keys 'train_loss', 'val_loss', 'val_miou', 'val_f1'
            save_name: output filename
        """
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        
        # Loss curve
        axes[0].plot(history['train_loss'], label='Train', marker='o', markersize=3)
        axes[0].plot(history['val_loss'], label='Val', marker='s', markersize=3)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # mIoU curve
        axes[1].plot([v * 100 for v in history['val_miou']], 
                    label='mIoU', color='purple', marker='o', markersize=3)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('mIoU (%)')
        axes[1].set_title('Mean IoU')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # F1 curve
        axes[2].plot([v * 100 for v in history['val_f1']], 
                    label='F1', color='red', marker='o', markersize=3)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('F1-Score (%)')
        axes[2].set_title('Macro F1-Score')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.plot_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved training curves: {save_path}")
    
    def plot_per_class_metrics(self, metrics: Dict, 
                              save_name_iou: str = "per_class_iou.png",
                              save_name_f1: str = "per_class_f1.png"):
        """
        Plot per-class IoU and F1-scores.
        
        Args:
            metrics: metrics dictionary
            save_name_iou: output filename for IoU plot
            save_name_f1: output filename for F1 plot
        """
        # Per-class IoU
        fig, ax = plt.subplots(figsize=(12, 6))
        iou_values = metrics['per_class_iou']
        colors = ['#e74c3c' if v < 0.3 else '#f39c12' if v < 0.5 else '#2ecc71'
                 for v in iou_values]
        bars = ax.bar(range(self.config.NUM_CLASSES), iou_values, 
                     color=colors, edgecolor='black', alpha=0.7)
        ax.axhline(y=metrics['mIoU'], color='blue', linestyle='--', 
                  linewidth=2, label=f"mIoU = {metrics['mIoU']:.3f}")
        ax.set_xlabel('Class')
        ax.set_ylabel('IoU')
        ax.set_title('Per-Class IoU (Test Set)')
        ax.set_xticks(range(self.config.NUM_CLASSES))
        ax.set_xticklabels(self.config.CLASS_NAMES, rotation=45, ha='right')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        save_path = self.plot_dir / save_name_iou
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved per-class IoU: {save_path}")
        
        # Per-class F1
        fig, ax = plt.subplots(figsize=(12, 6))
        f1_values = metrics['per_class_f1']
        colors = ['#e74c3c' if v < 0.3 else '#f39c12' if v < 0.5 else '#2ecc71'
                 for v in f1_values]
        bars = ax.bar(range(self.config.NUM_CLASSES), f1_values, 
                     color=colors, edgecolor='black', alpha=0.7)
        ax.axhline(y=metrics['macro_f1'], color='red', linestyle='--', 
                  linewidth=2, label=f"F1 Macro = {metrics['macro_f1']:.3f}")
        ax.set_xlabel('Class')
        ax.set_ylabel('F1-Score')
        ax.set_title('Per-Class F1-Score (Test Set)')
        ax.set_xticks(range(self.config.NUM_CLASSES))
        ax.set_xticklabels(self.config.CLASS_NAMES, rotation=45, ha='right')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        save_path = self.plot_dir / save_name_f1
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved per-class F1: {save_path}")
    
    def save_metrics_json(self, metrics: Dict, best_val_miou: float, 
                         epoch_times: List[float], 
                         save_name: str = "test_metrics.json"):
        """
        Save metrics to JSON file.
        
        Args:
            metrics: test metrics dictionary
            best_val_miou: best validation mIoU during training
            epoch_times: list of epoch durations
            save_name: output filename
        """
        data = {
            'test_metrics': {
                'OA': float(metrics['OA']),
                'mIoU': float(metrics['mIoU']),
                'macro_f1': float(metrics['macro_f1']),
                'per_class_iou': [float(v) for v in metrics['per_class_iou']],
                'per_class_f1': [float(v) for v in metrics['per_class_f1']],
                'per_class_precision': [float(v) for v in metrics['per_class_precision']],
                'per_class_recall': [float(v) for v in metrics['per_class_recall']],
            },
            'training_summary': {
                'best_val_miou': float(best_val_miou),
                'avg_epoch_time_sec': float(np.mean(epoch_times)),
                'total_training_time_sec': float(np.sum(epoch_times)),
            },
            'config': self.config.to_dict(),
            'class_names': self.config.CLASS_NAMES,
        }
        
        save_path = self.results_dir / save_name
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved metrics: {save_path}")
    
    def save_training_log(self, history: Dict, epoch_times: List[float], 
                         save_name: str = "training_log.csv"):
        """
        Save epoch-wise training log as CSV.
        
        Args:
            history: training history dictionary
            epoch_times: list of epoch durations
            save_name: output filename
        """
        save_path = self.results_dir / save_name
        
        with open(save_path, 'w') as f:
            f.write("epoch,train_loss,val_loss,oa,miou,f1,epoch_time_sec\n")
            
            for i in range(len(history['train_loss'])):
                f.write(f"{i+1},"
                       f"{history['train_loss'][i]:.6f},"
                       f"{history['val_loss'][i]:.6f},"
                       f"{history['val_oa'][i]:.6f},"
                       f"{history['val_miou'][i]:.6f},"
                       f"{history['val_f1'][i]:.6f},"
                       f"{epoch_times[i]:.1f}\n")
        
        logger.info(f"Saved training log: {save_path}")
    
    def save_all_results(self, predictions: np.ndarray, labels: np.ndarray,
                        history: Dict, best_val_miou: float, 
                        metrics: Dict, epoch_times: List[float]):
        """
        Save all results: plots, metrics, logs.
        
        Args:
            predictions: final test predictions
            labels: test labels
            history: training history
            best_val_miou: best validation mIoU
            metrics: final metrics dictionary
            epoch_times: list of epoch durations
        """
        logger.info("Saving all results...")
        
        self.plot_confusion_matrix(predictions, labels)
        self.plot_training_curves(history)
        self.plot_per_class_metrics(metrics)
        self.save_metrics_json(metrics, best_val_miou, epoch_times)
        self.save_training_log(history, epoch_times)
        
        logger.info("✓ All results saved successfully")
