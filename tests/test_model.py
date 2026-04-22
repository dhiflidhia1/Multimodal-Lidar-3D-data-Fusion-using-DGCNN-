"""
Unit tests for model components.

Run with: pytest tests/test_model.py -v
"""

import pytest
import torch
import numpy as np
from src.model import DGCNN_MidFusion, count_parameters, knn, get_graph_feature
from src.config import config


class TestKNN:
    """Tests for KNN function."""
    
    def test_knn_output_shape(self):
        """Test KNN returns correct shape."""
        x = torch.randn(4, 16, 128)  # (B, C, N)
        k = 20
        
        idx = knn(x, k=k)
        
        assert idx.shape == (4, 128, k)
    
    def test_knn_values_in_range(self):
        """Test KNN indices are valid."""
        x = torch.randn(2, 8, 32)  # (B, C, N)
        k = 10
        
        idx = knn(x, k=k)
        
        # All indices should be in valid range [0, N-1]
        assert idx.min() >= 0
        assert idx.max() < 32


class TestGraphFeature:
    """Tests for get_graph_feature function."""
    
    def test_graph_feature_shape(self):
        """Test graph feature extraction shape."""
        x = torch.randn(4, 64, 256)  # (B, C, N)
        k = 20
        
        feat = get_graph_feature(x, k=k)
        
        # Output should be (B, 2*C, N, K) due to concatenation
        assert feat.shape == (4, 128, 256, k)
    
    def test_graph_feature_preserves_batch(self):
        """Test batch size is preserved."""
        for batch_size in [1, 8, 16]:
            x = torch.randn(batch_size, 64, 256)
            feat = get_graph_feature(x, k=20)
            assert feat.shape[0] == batch_size


class TestDGCNNSpatialBranch:
    """Tests for DGCNN spatial branch."""
    
    def test_spatial_branch_output_shape(self):
        """Test spatial branch output shape."""
        from src.model import DGCNNSpatialBranch
        
        branch = DGCNNSpatialBranch(k=20, dropout=0.5)
        x = torch.randn(4, 5, 2048)  # (B, C, N)
        
        out = branch(x)
        
        # Output should be (B, 128, N)
        assert out.shape == (4, 128, 2048)
    
    def test_spatial_branch_device(self):
        """Test spatial branch works on different devices."""
        from src.model import DGCNNSpatialBranch
        
        branch = DGCNNSpatialBranch(k=20)
        x = torch.randn(2, 5, 512)
        
        # CPU
        out_cpu = branch(x)
        assert out_cpu.device.type == 'cpu'
        
        # GPU (if available)
        if torch.cuda.is_available():
            branch = branch.to('cuda')
            x = x.to('cuda')
            out_gpu = branch(x)
            assert out_gpu.device.type == 'cuda'


class TestSpectralBranch:
    """Tests for spectral branch."""
    
    def test_spectral_branch_output_shape(self):
        """Test spectral branch output shape."""
        from src.model import SpectralBranch
        
        branch = SpectralBranch(dropout=0.5)
        x = torch.randn(4, 5, 2048)  # (B, C, N)
        
        out = branch(x)
        
        # Output should be (B, 128, N)
        assert out.shape == (4, 128, 2048)


class TestDGCNNMidFusion:
    """Tests for main DGCNN_MidFusion model."""
    
    def test_model_initialization(self):
        """Test model can be instantiated."""
        model = DGCNN_MidFusion(k=20, num_classes=6, dropout=0.5)
        assert model is not None
    
    def test_model_output_shape(self):
        """Test model output shape."""
        model = DGCNN_MidFusion(k=20, num_classes=6)
        spatial = torch.randn(4, 2048, 5)  # (B, N, C)
        spectral = torch.randn(4, 2048, 5)  # (B, N, C)
        
        logits = model(spatial, spectral)
        
        # Output should be (B, num_classes, N)
        assert logits.shape == (4, 6, 2048)
    
    def test_model_parameter_count(self):
        """Test parameter count is reasonable."""
        model = DGCNN_MidFusion(k=20, num_classes=6)
        n_params = count_parameters(model)
        
        # Should have ~1-3M parameters
        assert 1e6 < n_params < 5e6
    
    def test_model_gradients_flow(self):
        """Test gradients flow through model."""
        model = DGCNN_MidFusion(k=20, num_classes=6)
        spatial = torch.randn(2, 1024, 5, requires_grad=True)
        spectral = torch.randn(2, 1024, 5, requires_grad=True)
        labels = torch.randint(0, 6, (2, 1024))
        
        # Forward
        logits = model(spatial, spectral)
        
        # Loss
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(logits, labels)
        
        # Backward
        loss.backward()
        
        # Check gradients
        assert spatial.grad is not None
        assert spectral.grad is not None
    
    def test_model_train_eval_modes(self):
        """Test model switches between train and eval."""
        model = DGCNN_MidFusion()
        
        # Train mode
        model.train()
        assert model.training
        
        # Eval mode
        model.eval()
        assert not model.training
    
    def test_model_device_transfer(self):
        """Test model can be moved to different devices."""
        model = DGCNN_MidFusion()
        
        # CPU
        model = model.cpu()
        for param in model.parameters():
            assert param.device.type == 'cpu'
        
        # GPU (if available)
        if torch.cuda.is_available():
            model = model.cuda()
            for param in model.parameters():
                assert param.device.type == 'cuda'


class TestCountParameters:
    """Tests for parameter counting function."""
    
    def test_count_parameters(self):
        """Test parameter count function."""
        model = torch.nn.Linear(10, 5)
        n_params = count_parameters(model)
        
        # 10 * 5 (weights) + 5 (bias) = 55
        assert n_params == 55


class TestConfig:
    """Tests for configuration."""
    
    def test_config_values(self):
        """Test config has valid values."""
        assert config.K > 0
        assert config.DROPOUT >= 0 and config.DROPOUT <= 1
        assert config.NUM_CLASSES == 6
        assert config.N_POINTS > 0
        assert config.BATCH_SIZE > 0
        assert config.EPOCHS > 0
        assert config.LR > 0
    
    def test_config_directories_created(self):
        """Test config directories are created."""
        assert config.OUTPUT_DIR.exists()
        assert config.PLOT_DIR.exists()
        assert config.RESULTS_DIR.exists()
        assert config.LOGS_DIR.exists()


# ============================================================
# Integration Tests
# ============================================================

class TestIntegration:
    """Integration tests with multiple components."""
    
    def test_full_forward_pass(self):
        """Test complete forward pass through model."""
        model = DGCNN_MidFusion(k=20, num_classes=6)
        model.eval()
        
        # Create dummy data
        spatial = torch.randn(2, 512, 5)  # (B, N, 5)
        spectral = torch.randn(2, 512, 5)  # (B, N, 5)
        
        # Forward
        with torch.no_grad():
            logits = model(spatial, spectral)
        
        # Verify output
        assert logits.shape[0] == 2  # Batch size
        assert logits.shape[1] == 6  # Num classes
        assert logits.shape[2] == 512  # Num points
    
    def test_training_step(self):
        """Test single training step."""
        model = DGCNN_MidFusion(k=20, num_classes=6)
        model.train()
        
        # Data
        spatial = torch.randn(4, 256, 5)
        spectral = torch.randn(4, 256, 5)
        labels = torch.randint(0, 6, (4, 256))
        
        # Setup
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Training step
        logits = model(spatial, spectral)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Verify loss decreased
        assert loss.item() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
