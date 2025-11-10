"""
Configuration Manager for XAI Analysis Pipeline.

Handles loading, validation, and management of configuration parameters
for the 3D CNN explainability analysis system.
"""

import os
import yaml
import shutil
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime
import hashlib
import json


class ConfigManager:
    """
    Manages configuration for XAI analysis pipeline.
    
    Supports loading from YAML files, validation of parameters,
    and saving configuration to output directories for reproducibility.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize ConfigManager with a configuration file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid YAML
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self._config_hash = self._compute_hash()
        
    def validate(self) -> bool:
        """
        Validate configuration parameters.
        
        Checks:
        - Required sections exist
        - File paths are valid
        - Numeric values are in acceptable ranges
        - Enum values are valid choices
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If validation fails with detailed error message
        """
        errors = []
        
        # Check required sections
        required_sections = ['model', 'data', 'gradcam', 'atlas', 'visualization', 'output']
        for section in required_sections:
            if section not in self.config:
                errors.append(f"Missing required section: {section}")
        
        if errors:
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(errors))
        
        # Validate model section
        if 'model' in self.config:
            model_config = self.config['model']
            
            if 'weights_dir' in model_config:
                weights_dir = Path(model_config['weights_dir'])
                if not weights_dir.exists():
                    errors.append(f"Model weights directory not found: {weights_dir}")
            
            if 'num_folds' in model_config:
                num_folds = model_config['num_folds']
                if not isinstance(num_folds, int) or num_folds < 1:
                    errors.append(f"num_folds must be positive integer, got: {num_folds}")
            
            if 'device' in model_config:
                device = model_config['device']
                valid_devices = ['cpu', 'cuda', 'cuda:0', 'cuda:1']
                if not any(device.startswith(d) for d in valid_devices):
                    errors.append(f"Invalid device: {device}")
        
        # Validate data section
        if 'data' in self.config:
            data_config = self.config['data']
            
            if 'patch_size' in data_config:
                patch_size = data_config['patch_size']
                if not isinstance(patch_size, list) or len(patch_size) != 3:
                    errors.append(f"patch_size must be list of 3 integers, got: {patch_size}")
            
            if 'target_voxel_size' in data_config:
                voxel_size = data_config['target_voxel_size']
                if not isinstance(voxel_size, list) or len(voxel_size) != 3:
                    errors.append(f"target_voxel_size must be list of 3 floats, got: {voxel_size}")
        
        # Validate gradcam section
        if 'gradcam' in self.config:
            gradcam_config = self.config['gradcam']
            
            if 'threshold_percentile' in gradcam_config:
                threshold = gradcam_config['threshold_percentile']
                if not isinstance(threshold, (int, float)) or not (0 <= threshold <= 100):
                    errors.append(f"threshold_percentile must be between 0 and 100, got: {threshold}")
            
            if 'aggregation_method' in gradcam_config:
                method = gradcam_config['aggregation_method']
                valid_methods = ['mean', 'max', 'weighted']
                if method not in valid_methods:
                    errors.append(f"aggregation_method must be one of {valid_methods}, got: {method}")
        
        # Validate atlas section
        if 'atlas' in self.config:
            atlas_config = self.config['atlas']
            
            if 'path' in atlas_config:
                atlas_path = Path(atlas_config['path'])
                if not atlas_path.exists():
                    errors.append(f"Atlas file not found: {atlas_path}")
            
            if 'labels_path' in atlas_config:
                labels_path = Path(atlas_config['labels_path'])
                if not labels_path.exists():
                    errors.append(f"Atlas labels file not found: {labels_path}")
        
        # Validate visualization section
        if 'visualization' in self.config:
            viz_config = self.config['visualization']
            
            if 'alpha' in viz_config:
                alpha = viz_config['alpha']
                if not isinstance(alpha, (int, float)) or not (0 <= alpha <= 1):
                    errors.append(f"alpha must be between 0 and 1, got: {alpha}")
            
            if 'display_mode' in viz_config:
                mode = viz_config['display_mode']
                valid_modes = ['ortho', 'x', 'y', 'z', 'xz', 'yz', 'xy']
                if mode not in valid_modes:
                    errors.append(f"display_mode must be one of {valid_modes}, got: {mode}")
        
        # Validate output section
        if 'output' in self.config:
            output_config = self.config['output']
            
            if 'base_dir' in output_config:
                base_dir = Path(output_config['base_dir'])
                # Create if doesn't exist
                base_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate batch section if present
        if 'batch' in self.config:
            batch_config = self.config['batch']
            
            if 'max_workers' in batch_config:
                max_workers = batch_config['max_workers']
                if not isinstance(max_workers, int) or max_workers < 1:
                    errors.append(f"max_workers must be positive integer, got: {max_workers}")
        
        if errors:
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(errors))
        
        return True
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., 'model.device')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
            
        Examples:
            >>> config.get('model.device')
            'cuda:0'
            >>> config.get('gradcam.threshold_percentile', 95.0)
            95.0
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def save_to_output(self, output_dir: str) -> str:
        """
        Save configuration to output directory for reproducibility.
        
        Creates a copy of the configuration file with metadata including
        timestamp and config hash.
        
        Args:
            output_dir: Directory to save configuration
            
        Returns:
            Path to saved configuration file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create config copy with metadata
        config_with_metadata = {
            'metadata': {
                'saved_at': datetime.now().isoformat(),
                'config_hash': self._config_hash,
                'original_path': str(self.config_path.absolute())
            },
            'configuration': self.config
        }
        
        # Save as YAML
        config_file = output_path / 'config.yaml'
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_with_metadata, f, default_flow_style=False, allow_unicode=True)
        
        # Also save as JSON for easier parsing
        json_file = output_path / 'config.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(config_with_metadata, f, indent=2, ensure_ascii=False)
        
        return str(config_file)
    
    def _compute_hash(self) -> str:
        """
        Compute hash of configuration for versioning.
        
        Returns:
            MD5 hash of configuration content
        """
        config_str = yaml.dump(self.config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def get_hash(self) -> str:
        """
        Get configuration hash.
        
        Returns:
            MD5 hash of configuration
        """
        return self._config_hash
    
    def to_dict(self) -> Dict:
        """
        Export configuration as dictionary.
        
        Returns:
            Configuration dictionary
        """
        return self.config.copy()
    
    def __repr__(self) -> str:
        """String representation of ConfigManager."""
        return f"ConfigManager(config_path='{self.config_path}', hash='{self._config_hash[:8]}')"
