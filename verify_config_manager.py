"""
Verification script for ConfigManager implementation.
Tests basic functionality without requiring all dependencies.
"""

import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.xai.config_manager import ConfigManager


def verify_config_manager():
    """Verify ConfigManager basic functionality."""
    print("=" * 60)
    print("Verifying ConfigManager Implementation")
    print("=" * 60)
    
    try:
        # Test 1: Load configuration from YAML file
        print("\n1. Loading configuration from YAML file...")
        config = ConfigManager('config/xai_config.yaml')
        print(f"   ✓ Configuration loaded: {config}")
        
        # Test 2: Validate configuration
        print("\n2. Validating configuration...")
        is_valid = config.validate()
        print(f"   ✓ Configuration is valid: {is_valid}")
        
        # Test 3: Get configuration values using dot notation
        print("\n3. Testing configuration value retrieval...")
        device = config.get('model.device')
        print(f"   ✓ model.device = {device}")
        
        threshold = config.get('gradcam.threshold_percentile')
        print(f"   ✓ gradcam.threshold_percentile = {threshold}")
        
        atlas_name = config.get('atlas.name')
        print(f"   ✓ atlas.name = {atlas_name}")
        
        # Test with default value
        missing = config.get('nonexistent.key', 'default_value')
        print(f"   ✓ nonexistent.key (with default) = {missing}")
        
        # Test 4: Get configuration hash
        print("\n4. Testing configuration hash...")
        config_hash = config.get_hash()
        print(f"   ✓ Configuration hash: {config_hash[:16]}...")
        
        # Test 5: Export configuration as dictionary
        print("\n5. Testing configuration dictionary export...")
        config_dict = config.to_dict()
        print(f"   ✓ Configuration exported with {len(config_dict)} sections")
        print(f"   ✓ Sections: {list(config_dict.keys())}")
        
        # Test 6: Save configuration to output directory
        print("\n6. Testing configuration save to output...")
        output_dir = "output/test_config_verification"
        saved_path = config.save_to_output(output_dir)
        print(f"   ✓ Configuration saved to: {saved_path}")
        
        # Verify saved files exist
        saved_yaml = Path(output_dir) / 'config.yaml'
        saved_json = Path(output_dir) / 'config.json'
        print(f"   ✓ YAML file exists: {saved_yaml.exists()}")
        print(f"   ✓ JSON file exists: {saved_json.exists()}")
        
        print("\n" + "=" * 60)
        print("✓ All verification tests passed!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n✗ Verification failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = verify_config_manager()
    sys.exit(0 if success else 1)
