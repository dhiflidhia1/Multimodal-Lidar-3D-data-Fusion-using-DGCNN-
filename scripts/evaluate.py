#!/usr/bin/env python3
"""
Evaluation script for DGCNN Mid-Level Fusion model.
Evaluate a trained model on test set without retraining.

Usage:
    python scripts/evaluate.py --model-path outputs/checkpoints/best_model.pth
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src import config, create_dataloaders, create_model, create_criterion
from src.training import Trainer
from src.visualization import ResultsVisualizer
from src.metrics import print_metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_model(model_path: str):
    """
    Evaluate model on test set.
    
    Args:
        model_path: path to saved model weights
    """
    logger.info("=" * 70)
    logger.info("EVALUATION")
    logger.info("=" * 70)
    
    # Check device
    logger.info(f"Device: {config.DEVICE}")
    
    # Load data
    logger.info("\nLoading data...")
    _, _, test_loader = create_dataloaders(str(config.DATA_PATH), config)
    logger.info(f"Test batches: {len(test_loader)}")
    
    # Create model
    logger.info("\nCreating model...")
    model = create_model(config)
    
    # Load weights
    logger.info(f"Loading model from: {model_path}")
    model.load_state_dict(
        torch.load(model_path, map_location=config.DEVICE, weights_only=True)
    )
    
    # Create criterion and trainer
    criterion = create_criterion(config, config.DEVICE)
    trainer = Trainer(model, criterion, None, config, config.DEVICE)
    
    # Evaluate
    logger.info("\nEvaluating on test set...")
    test_loss, test_metrics, test_preds, test_labels = trainer.evaluate(test_loader)
    
    # Print results
    logger.info("\n" + "=" * 70)
    logger.info("TEST RESULTS")
    logger.info("=" * 70)
    logger.info(f"Test Loss: {test_loss:.4f}")
    print_metrics(test_metrics, config.CLASS_NAMES, prefix="Test")
    
    # Save visualizations
    logger.info("\nSaving visualizations...")
    visualizer = ResultsVisualizer(config)
    visualizer.plot_confusion_matrix(test_preds, test_labels, 
                                     save_name="test_confusion_matrix.png")
    visualizer.save_metrics_json(test_metrics, best_val_miou=0.0, 
                                epoch_times=[0], 
                                save_name="test_results.json")
    
    logger.info("\n✓ Evaluation complete. Results saved to outputs/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DGCNN model")
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to model checkpoint')
    args = parser.parse_args()
    
    evaluate_model(args.model_path)
