"""
3D ResNet-10 Mini-CNN for ROI Feature Extraction
用於 ROI 特徵提取的 3D ResNet-10 Mini-CNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock3D(nn.Module):
    """3D ResNet Basic Block"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock3D, self).__init__()
        
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, 
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        
        self.conv2 = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet3D_Mini(nn.Module):
    """
    3D ResNet-10 Mini-CNN for Feature Extraction
    
    Architecture:
    - Input: (B, 1, D, H, W) - Single modality 3D patch
    - Output: (B, feature_dim) - Feature vector
    
    Parameters:
    -----------
    in_channels : int
        Number of input channels (1 for single modality)
    feature_dim : int
        Output feature dimension (default: 64)
    block_config : list
        Number of blocks in each layer (default: [1, 1, 1, 1] for ResNet-10)
    initial_filters : int
        Number of filters in first conv layer (default: 32)
    """
    
    def __init__(
        self, 
        in_channels=1, 
        feature_dim=64,
        block_config=[1, 1, 1, 1],
        initial_filters=32
    ):
        super(ResNet3D_Mini, self).__init__()
        
        self.in_channels = initial_filters
        
        # Initial convolution
        self.conv1 = nn.Conv3d(
            in_channels, initial_filters,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm3d(initial_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(BasicBlock3D, initial_filters, block_config[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock3D, initial_filters * 2, block_config[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock3D, initial_filters * 4, block_config[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock3D, initial_filters * 8, block_config[3], stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Feature projection
        self.fc = nn.Linear(initial_filters * 8, feature_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, block, out_channels, num_blocks, stride=1):
        """Create a ResNet layer with multiple blocks"""
        downsample = None
        
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.in_channels, out_channels * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm3d(out_channels * block.expansion),
            )
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (B, 1, D, H, W)
        
        Returns:
        --------
        features : torch.Tensor
            Feature vector of shape (B, feature_dim)
        """
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Feature projection
        features = self.fc(x)
        
        return features


class MultiModalFeatureExtractor(nn.Module):
    """
    Multi-modal Feature Extractor using 3 independent ResNet3D Mini-CNNs
    
    Architecture:
    - Mini-CNN_T1: Extract features from T1 patches
    - Mini-CNN_FLAIR: Extract features from T2-FLAIR patches
    - Mini-CNN_DWI: Extract features from DWI/FA patches
    
    For each subject:
    - Input: 116 patches × 3 modalities
    - Output: 22,104-dim feature vector (116 × 3 × 64)
    """
    
    def __init__(self, feature_dim=64, initial_filters=32):
        super(MultiModalFeatureExtractor, self).__init__()
        
        # Three independent Mini-CNNs
        self.mini_cnn_t1 = ResNet3D_Mini(
            in_channels=1,
            feature_dim=feature_dim,
            initial_filters=initial_filters
        )
        
        self.mini_cnn_flair = ResNet3D_Mini(
            in_channels=1,
            feature_dim=feature_dim,
            initial_filters=initial_filters
        )
        
        self.mini_cnn_dwi = ResNet3D_Mini(
            in_channels=1,
            feature_dim=feature_dim,
            initial_filters=initial_filters
        )
    
    def forward(self, t1_patches, flair_patches, dwi_patches):
        """
        Extract features from all ROI patches
        
        Parameters:
        -----------
        t1_patches : torch.Tensor
            T1 patches of shape (B, N_ROI, 1, D, H, W)
        flair_patches : torch.Tensor
            FLAIR patches of shape (B, N_ROI, 1, D, H, W)
        dwi_patches : torch.Tensor
            DWI patches of shape (B, N_ROI, 1, D, H, W)
        
        Returns:
        --------
        features : torch.Tensor
            Concatenated feature vector of shape (B, N_ROI * 3 * feature_dim)
        """
        batch_size, n_rois = t1_patches.shape[:2]
        
        # Reshape: (B, N_ROI, 1, D, H, W) -> (B*N_ROI, 1, D, H, W)
        t1_flat = t1_patches.view(-1, *t1_patches.shape[2:])
        flair_flat = flair_patches.view(-1, *flair_patches.shape[2:])
        dwi_flat = dwi_patches.view(-1, *dwi_patches.shape[2:])
        
        # Extract features
        t1_features = self.mini_cnn_t1(t1_flat)  # (B*N_ROI, feature_dim)
        flair_features = self.mini_cnn_flair(flair_flat)
        dwi_features = self.mini_cnn_dwi(dwi_flat)
        
        # Reshape back: (B*N_ROI, feature_dim) -> (B, N_ROI, feature_dim)
        t1_features = t1_features.view(batch_size, n_rois, -1)
        flair_features = flair_features.view(batch_size, n_rois, -1)
        dwi_features = dwi_features.view(batch_size, n_rois, -1)
        
        # Concatenate: (B, N_ROI, feature_dim * 3)
        features = torch.cat([t1_features, flair_features, dwi_features], dim=2)
        
        # Flatten: (B, N_ROI * feature_dim * 3)
        features = features.view(batch_size, -1)
        
        return features


def test_model():
    """Test the model architecture"""
    print("Testing 3D ResNet-10 Mini-CNN...")
    
    # Test single Mini-CNN
    model = ResNet3D_Mini(in_channels=1, feature_dim=64, initial_filters=32)
    x = torch.randn(2, 1, 32, 32, 32)  # (B, C, D, H, W)
    
    features = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Expected: (2, 64)")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test multi-modal extractor
    print("\n" + "="*60)
    print("Testing Multi-Modal Feature Extractor...")
    
    multi_model = MultiModalFeatureExtractor(feature_dim=64, initial_filters=32)
    
    # Simulate 116 ROI patches for 3 modalities
    t1_patches = torch.randn(2, 116, 1, 32, 32, 32)
    flair_patches = torch.randn(2, 116, 1, 32, 32, 32)
    dwi_patches = torch.randn(2, 116, 1, 32, 32, 32)
    
    features = multi_model(t1_patches, flair_patches, dwi_patches)
    print(f"Input shapes: (2, 116, 1, 32, 32, 32) × 3 modalities")
    print(f"Output shape: {features.shape}")
    print(f"Expected: (2, 22104) = 2 × (116 ROIs × 3 modalities × 64 features)")
    
    # Count total parameters
    total_params = sum(p.numel() for p in multi_model.parameters())
    print(f"\nTotal parameters (3 Mini-CNNs): {total_params:,}")
    
    print("\n[OK] Model test passed!")


if __name__ == "__main__":
    test_model()
