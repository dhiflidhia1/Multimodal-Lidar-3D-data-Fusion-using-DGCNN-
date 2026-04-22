#!/usr/bin/env python3
"""
Main training script for DGCNN Mid-Level Fusion model.

Usage:
    python scripts/train.py
"""

import os
import sys
import argparse
import logging
import time
import warnings
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

# Import project modules
from src import (
    config, create_dataloaders, create_model, 
    create_optimizer, create_criterion, Trainer, 
    ResultsVisualizer, print_metrics
)

warnings.filterwarnings("ignore")

# ============================================================
# LOGGING SETUP
# ============================================================

def setup_logging(config):
    """Configure logging."""
    log_dir = config.LOGS_DIR
    log_file = log_dir / "training.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


logger = setup_logging(config)

# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def main(args):
    """Main training loop."""
    
    logger.info("=" * 70)
    logger.info("DGCNN MID-LEVEL FUSION - FRACTAL DATASET")
    logger.info("=" * 70)
    
    # Print environment info
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Device: {config.DEVICE}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("Running on CPU only")
    
    # Print config
    config.print_summary()
    
    # ============================================================
    # DATA LOADING
    # ============================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("LOADING DATA")
    logger.info("=" * 70)
    
    data_path = str(config.DATA_PATH)
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        logger.error("Please ensure the data file is at: data/fractal_data_v2_filtered.npy")
        return
    
    train_loader, val_loader, test_loader = create_dataloaders(data_path, config)
    
    # ============================================================
    # MODEL SETUP
    # ============================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("INITIALIZING MODEL")
    logger.info("=" * 70)
    
    model = create_model(config)
    
    # ============================================================
    # OPTIMIZER & LOSS
    # ============================================================
    
    optimizer = create_optimizer(model, config)
    criterion = create_criterion(config, config.DEVICE)
    
    logger.info("Created optimizer: Adam")
    logger.info(f"Learning rate: {config.LR}")
    logger.info(f"Weight decay: {config.WEIGHT_DECAY}")
    
    # ============================================================
    # TRAINER & VISUALIZER
    # ============================================================
    
    trainer = Trainer(model, criterion, optimizer, config, config.DEVICE)
    visualizer = ResultsVisualizer(config)
    
    # ============================================================
    # TRAINING LOOP
    # ============================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING")
    logger.info("=" * 70)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_miou': [],
        'val_miou': [],
        'train_f1': [],
        'val_f1': [],
        'val_oa': [],
        'epoch_times': []
    }
    
    best_miou = 0.0
    best_model_path = config.OUTPUT_DIR / "best_model.pth"
    
    for epoch in range(1, config.EPOCHS + 1):
        epoch_start = time.time()
        
        # Training
        train_loss, train_metrics = trainer.train_one_epoch(train_loader)
        torch.cuda.empty_cache()
        
        # Validation
        val_loss, val_metrics, _, _ = trainer.evaluate(val_loader)
        torch.cuda.empty_cache()
        
        # Step scheduler
        trainer.step_scheduler()
        
        # Track history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_miou'].append(train_metrics['mIoU'])
        history['val_miou'].append(val_metrics['mIoU'])
        history['train_f1'].append(train_metrics['macro_f1'])
        history['val_f1'].append(val_metrics['macro_f1'])
        history['val_oa'].append(val_metrics['OA'])
        
        epoch_time = time.time() - epoch_start
        history['epoch_times'].append(epoch_time)
        
        # Log epoch
        remaining_epochs = config.EPOCHS - epoch
        remaining_time = epoch_time * remaining_epochs / 3600
        
        logger.info(
            f"\nEpoch {epoch:3d}/{config.EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"OA: {val_metrics['OA']*100:.1f}% | "
            f"mIoU: {val_metrics['mIoU']*100:.1f}% | "
            f"F1: {val_metrics['macro_f1']*100:.1f}%"
        )
        
        logger.info(f"Time: {epoch_time:.0f}s | Remaining: {remaining_time:.2f}h")
        
        # Per-class metrics
        logger.info("Per-class IoU & F1:")
        for name, iou, f1 in zip(config.CLASS_NAMES, 
                                val_metrics['per_class_iou'], 
                                val_metrics['per_class_f1']):
            logger.info(f"  {name:<20} - IoU: {iou*100:6.1f}% | F1: {f1*100:6.1f}%")
        
        # Save best model
        if val_metrics['mIoU'] > best_miou:
            best_miou = val_metrics['mIoU']
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"✓ New best model saved (mIoU: {best_miou*100:.2f}%)")
        
        # Save intermediate plots every N epochs
        if epoch % 10 == 0 or epoch == 1:
            visualizer.plot_training_curves(
                history, 
                save_name=f"training_curves_epoch{epoch}.png"
            )
    
    # ============================================================
    # FINAL EVALUATION
    # ============================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("FINAL EVALUATION ON TEST SET")
    logger.info("=" * 70)
    
    # Load best model
    model.load_state_dict(
        torch.load(best_model_path, map_location=config.DEVICE, weights_only=True)
    )
    
    # Test
    test_loss, test_metrics, test_preds, test_labels = trainer.evaluate(test_loader)
    
    logger.info(f"\nTest Loss: {test_loss:.4f}")
    logger.info(f"Test OA: {test_metrics['OA']*100:.2f}%")
    logger.info(f"Test mIoU: {test_metrics['mIoU']*100:.2f}%")
    logger.info(f"Test F1 Macro: {test_metrics['macro_f1']*100:.2f}%")
    
    # Per-class results
    logger.info(f"\n{'Class':<20}  {'IoU':>8}  {'F1':>8}  {'Precision':>10}  {'Recall':>8}")
    logger.info("─" * 65)
    for name, iou, f1, prec, rec in zip(
        config.CLASS_NAMES,
        test_metrics['per_class_iou'],
        test_metrics['per_class_f1'],
        test_metrics['per_class_precision'],
        test_metrics['per_class_recall']
    ):
        logger.info(f"{name:<20}  {iou*100:>7.2f}%  {f1*100:>7.2f}%  "
                   f"{prec*100:>9.2f}%  {rec*100:>7.2f}%")
    
    # ============================================================
    # SAVE ALL RESULTS
    # ============================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("SAVING RESULTS")
    logger.info("=" * 70)
    
    avg_time = float(np.mean(history['epoch_times']))
    visualizer.save_all_results(
        test_preds, test_labels, history, best_miou, 
        test_metrics, history['epoch_times']
    )
    
    # ============================================================
    # SUMMARY
    # ============================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Best Val mIoU: {best_miou*100:.2f}%")
    logger.info(f"Test OA:       {test_metrics['OA']*100:.2f}%")
    logger.info(f"Test mIoU:     {test_metrics['mIoU']*100:.2f}%")
    logger.info(f"Test F1 Macro: {test_metrics['macro_f1']*100:.2f}%")
    logger.info(f"Avg Time/Epoch: {avg_time:.0f}s ({avg_time/60:.1f}m)")
    logger.info(f"Total Time:    {sum(history['epoch_times'])/3600:.2f}h")
    logger.info("=" * 70)


# ============================================================
# ARGUMENT PARSING
# ============================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train DGCNN Mid-Level Fusion model"
    )
    
    # Data arguments
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path to data .npy file')
    
    # Model arguments
    parser.add_argument('--k', type=int, default=config.K,
                       help='Number of KNN neighbors')
    parser.add_argument('--dropout', type=float, default=config.DROPOUT,
                       help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=config.EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=config.LR,
                       help='Learning rate')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Override config if arguments provided
    if args.data_path:
        config.DATA_PATH = args.data_path
    if args.k != config.K:
        config.K = args.k
    if args.dropout != config.DROPOUT:
        config.DROPOUT = args.dropout
    if args.epochs != config.EPOCHS:
        config.EPOCHS = args.epochs
    if args.batch_size != config.BATCH_SIZE:
        config.BATCH_SIZE = args.batch_size
    if args.lr != config.LR:
        config.LR = args.lr
    
    main(args)
