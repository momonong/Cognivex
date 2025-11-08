"""
Quick test script to verify ConfigManager functionality.
"""

import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.xai.config_manager import ConfigManager


def test_config_manager():
    """Test ConfigManager basic functionality."""
    print("=" * 60)
    print("Testing ConfigManager")
    print("=" * 60)
    
    # Test 1: Load default configuration
    print("\n1. Testing default configuration loading...")
    config = ConfigManager()
    print(f"   ✓ ConfigManager created: {config}")
    print(f"   ✓ Sections: {list(config.config.keys())}")
    
    # Test 2: Load from YAML file
    print("\n2. Testing YAML configuration loading...")
    config_yaml = ConfigManager('config/xai_config.yaml')
    print(f"   ✓ Configuration loaded from: {config_yaml.config_path}")
    
    # Test 3: Get configuration values
    print("\n3. Testing configuration value retrieval...")
    device = config_yaml.get('model.device')
    print(f"   ✓ model.device = {device}")
    
    threshold = config_yaml.get('gradcam.threshold_percentile')
    print(f"   ✓ gradcam.threshold_percentile = {threshold}")
    
    atlas_name = config_yaml.get('atlas.name')
    print(f"   ✓ atlas.name = {atlas_name}")
    
    # Test 4: Set configuration values
    print("\n4. Testing configuration value setting...")
    config_yaml.set('model.device', 'cpu')
    new_device = config_yaml.get('model.device')
    print(f"   ✓ Updated model.device = {new_device}")
    
    # Test 5: Validate configuration
    print("\n5. Testing configuration validation...")
    is_valid = config_yaml.validate()
    if is_valid:
        print(f"   ✓ Configuration is valid")
    else:
        print(f"   ✗ Configuration validation failed:")
        for error in config_yaml.validation_errors:
            print(f"     - {error}")
    
    # Test 6: Save configuration to output directory
    print("\n6. Testing configuration save to output...")
    output_dir = "output/test_config"
    saved_path = config_yaml.save_to_output(output_dir)
    print(f"   ✓ Configuration saved to: {saved_path}")
    
    # Test 7: Get configuration as dictionary
    print("\n7. Testing configuration dictionary export...")
    config_dict = config_yaml.to_dict()
    print(f"   ✓ Configuration exported with {len(config_dict)} sections")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == '__main__':
    try:
        test_config_manager()
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
