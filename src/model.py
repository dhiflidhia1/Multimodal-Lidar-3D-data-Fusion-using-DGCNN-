"""
Model architecture: DGCNN with mid-level fusion.
Combines spatial (LIDAR) and spectral (RGB-IR) features.
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Compute k-nearest neighbors in feature space.
    
    Args:
        x: (B, C, N) - batch, channels, points
        k: number of neighbors
    
    Returns:
        idx: (B, N, K) - indices of k-nearest neighbors
    """
    # Compute pairwise distances using ||x||^2 - 2x^T y + ||y||^2
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  # (B, N, N)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)       # (B, 1, N)
    dist = -xx - inner - xx.transpose(2, 1)           # (B, N, N)
    return dist.topk(k=k, dim=-1)[1]


def get_graph_feature(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Extract graph features using k-nearest neighbors.
    
    Args:
        x: (B, C, N) - input features
        k: number of neighbors
    
    Returns:
        feat: (B, 2C, N, K) - edge features and center features
    """
    batch_size, channels, n_points = x.size()
    
    # Get KNN indices
    idx = knn(x, k=k)  # (B, N, K)
    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * n_points
    idx = (idx + idx_base).view(-1)
    
    # Gather neighbor features
    x_t = x.transpose(2, 1).contiguous()  # (B, N, C)
    feat = x_t.view(batch_size * n_points, -1)[idx].view(batch_size, n_points, k, channels)
    
    # Replicate center features
    x_rep = x_t.view(batch_size, n_points, 1, channels).repeat(1, 1, k, 1)
    
    # Concatenate edge vectors [neighbor - center, center]
    return torch.cat([feat - x_rep, x_rep], dim=3).permute(0, 3, 1, 2).contiguous()


class DGCNNSpatialBranch(nn.Module):
    """DGCNN branch for spatial (LIDAR) features."""
    
    def __init__(self, k: int, dropout: float = 0.5):
        """
        Args:
            k: number of neighbors
            dropout: dropout rate
        """
        super().__init__()
        self.k = k
        
        # Sequential graph convolution layers
        self.layer1 = nn.Sequential(
            nn.Conv2d(10, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 5, N) - spatial features (X, Y, Z, ReturnNum, NumReturns)
        
        Returns:
            out: (B, 128, N) - fused spatial features
        """
        # Layer 1: Initial graph convolution
        x = get_graph_feature(x, self.k)  # (B, 10, N, K)
        x = self.layer1(x)                # (B, 64, N, K)
        x = x.max(dim=-1)[0]              # (B, 64, N)
        
        # Layer 2
        x = get_graph_feature(x, self.k)  # (B, 128, N, K)
        x = self.layer2(x)                # (B, 64, N, K)
        x = x.max(dim=-1)[0]              # (B, 64, N)
        
        # Layer 3
        x = get_graph_feature(x, self.k)  # (B, 128, N, K)
        x = self.layer3(x)                # (B, 128, N, K)
        x = x.max(dim=-1)[0]              # (B, 128, N)
        
        # Layer 4
        x = get_graph_feature(x, self.k)  # (B, 256, N, K)
        x = self.layer4(x)                # (B, 128, N, K)
        x = x.max(dim=-1)[0]              # (B, 128, N)
        
        return x


class SpectralBranch(nn.Module):
    """Sequential branch for spectral (RGB-IR) features."""
    
    def __init__(self, dropout: float = 0.5):
        """Args: dropout: dropout rate"""
        super().__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv1d(5, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2)
        )
        
        self.layer4 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 5, N) - spectral features (R, G, B, IR, NDVI)
        
        Returns:
            out: (B, 128, N) - spectral features
        """
        x = self.layer1(x)  # (B, 64, N)
        x = self.layer2(x)  # (B, 64, N)
        x = self.layer3(x)  # (B, 128, N)
        x = self.layer4(x)  # (B, 128, N)
        return x


class DGCNN_MidFusion(nn.Module):
    """
    DGCNN with mid-level fusion of spatial and spectral modalities.
    
    Architecture:
    1. Spatial branch: DGCNN on point coordinates + geometric features
    2. Spectral branch: Sequential convolutions on RGB-IR features
    3. Mid-level fusion: Concatenate intermediate features
    4. Global aggregation: Max pooling and concat
    5. Classification head: FC layers for per-point predictions
    """
    
    def __init__(self, k: int = 16, num_classes: int = 6, dropout: float = 0.5):
        """
        Args:
            k: number of neighbors for graph construction
            num_classes: number of semantic classes
            dropout: dropout rate
        """
        super().__init__()
        self.k = k
        
        # Branches
        self.spatial_branch = DGCNNSpatialBranch(k=k, dropout=dropout)
        self.spectral_branch = SpectralBranch(dropout=dropout)
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2)
        )
        
        # Global aggregation
        self.global_mlp = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2)
        )
        
        # Classification head
        self.head = nn.Sequential(
            nn.Conv1d(768, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Conv1d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, num_classes, kernel_size=1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, spatial: torch.Tensor, spectral: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            spatial: (B, N, 5) - spatial features
            spectral: (B, N, 5) - spectral features
        
        Returns:
            logits: (B, num_classes, N) - per-point class logits
        """
        batch_size, n_points, _ = spatial.shape
        
        # Transpose to (B, C, N) for convolution
        spatial = spatial.transpose(2, 1)   # (B, 5, N)
        spectral = spectral.transpose(2, 1)  # (B, 5, N)
        
        # Extract features from both branches
        spatial_feat = self.spatial_branch(spatial)    # (B, 128, N)
        spectral_feat = self.spectral_branch(spectral)  # (B, 128, N)
        
        # Mid-level fusion
        fused = self.fusion(torch.cat([spatial_feat, spectral_feat], dim=1))  # (B, 256, N)
        
        # Global context
        global_feat = self.global_mlp(fused).max(dim=-1)[0]  # (B, 512)
        global_feat = global_feat.unsqueeze(-1).expand(-1, -1, n_points)  # (B, 512, N)
        
        # Classify
        logits = self.head(torch.cat([fused, global_feat], dim=1))  # (B, num_classes, N)
        
        return logits


def count_parameters(model: nn.Module) -> int:
    """Count total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model(config) -> nn.Module:
    """
    Create and initialize model.
    
    Args:
        config: Config object
    
    Returns:
        model on the specified device
    """
    model = DGCNN_MidFusion(
        k=config.K,
        num_classes=config.NUM_CLASSES,
        dropout=config.DROPOUT
    ).to(config.DEVICE)
    
    n_params = count_parameters(model)
    logger.info(f"Created DGCNN_MidFusion with {n_params / 1e6:.2f}M parameters")
    
    return model
