# tests/test_activation_extractor.py

"""
測試 ActivationExtractor 類別的功能。
驗證 activation 和 gradient 擷取、儲存和載入的正確性。
"""

import torch
import torch.nn as nn
import os
import sys
import tempfile
import shutil

# 加入專案根目錄到路徑
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from app.core.xai.activation_extractor import ActivationExtractor
from app.core.cnn_3d.model import Simple3DCNN_InstanceNorm


def test_activation_extractor_basic():
    """測試基本的 activation 和 gradient 擷取功能。"""
    print("\n=== 測試 1: 基本 Activation 擷取 ===")
    
    # 建立模型
    model = Simple3DCNN_InstanceNorm(in_channels=1, num_classes=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # 建立 ActivationExtractor
    target_layers = ['block4', 'block3']
    extractor = ActivationExtractor(model, target_layers, device)
    
    # 註冊 hooks
    extractor.register_hooks()
    
    # 建立測試輸入 (batch_size=1, channels=1, H=128, W=128, D=128)
    test_input = torch.randn(1, 1, 128, 128, 128).to(device)
    
    # 擷取 activations 和 gradients
    results = extractor.extract(test_input, target_class=1, subject_id="test_subject")
    
    # 驗證結果
    assert len(results) == 2, f"應該擷取 2 個層，但得到 {len(results)} 個"
    
    for layer_name in target_layers:
        assert layer_name in results, f"缺少層 {layer_name}"
        assert 'activation' in results[layer_name], f"層 {layer_name} 缺少 activation"
        assert 'gradient' in results[layer_name], f"層 {layer_name} 缺少 gradient"
        assert 'metadata' in results[layer_name], f"層 {layer_name} 缺少 metadata"
        
        # 驗證 metadata
        metadata = results[layer_name]['metadata']
        assert metadata['subject_id'] == 'test_subject'
        assert metadata['layer_name'] == layer_name
        assert metadata['target_class'] == 1
        assert 'timestamp' in metadata
        assert 'activation_shape' in metadata
        assert 'gradient_shape' in metadata
        
        print(f"✅ 層 {layer_name}:")
        print(f"   Activation shape: {results[layer_name]['activation'].shape}")
        print(f"   Gradient shape: {results[layer_name]['gradient'].shape}")
    
    # 移除 hooks
    extractor.remove_hooks()
    
    print("✅ 測試 1 通過！\n")
    return results


def test_save_and_load():
    """測試儲存和載入功能。"""
    print("\n=== 測試 2: 儲存和載入功能 ===")
    
    # 建立模型
    model = Simple3DCNN_InstanceNorm(in_channels=1, num_classes=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # 建立 ActivationExtractor
    target_layers = ['block4']
    extractor = ActivationExtractor(model, target_layers, device)
    extractor.register_hooks()
    
    # 建立測試輸入
    test_input = torch.randn(1, 1, 128, 128, 128).to(device)
    
    # 擷取資料
    results = extractor.extract(test_input, target_class=0, subject_id="save_test")
    
    # 建立臨時目錄
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 儲存資料
        print(f"儲存到: {temp_dir}")
        extractor.save_to_disk(results, temp_dir, subject_id="save_test")
        
        # 驗證檔案存在
        expected_file = os.path.join(temp_dir, "save_test_block4_activations.pt")
        assert os.path.exists(expected_file), f"檔案不存在: {expected_file}"
        
        # 載入資料
        loaded_data = extractor.load_from_disk(expected_file)
        
        # 驗證載入的資料
        assert 'activation' in loaded_data
        assert 'gradient' in loaded_data
        assert 'metadata' in loaded_data
        
        # 驗證資料內容相同
        original_activation = results['block4']['activation']
        loaded_activation = loaded_data['activation']
        
        assert torch.allclose(original_activation, loaded_activation), "載入的 activation 與原始資料不符"
        
        original_gradient = results['block4']['gradient']
        loaded_gradient = loaded_data['gradient']
        
        assert torch.allclose(original_gradient, loaded_gradient), "載入的 gradient 與原始資料不符"
        
        print("✅ 資料儲存和載入驗證成功！")
        print(f"   原始 activation shape: {original_activation.shape}")
        print(f"   載入 activation shape: {loaded_activation.shape}")
        print(f"   Metadata: {loaded_data['metadata']}")
        
    finally:
        # 清理臨時目錄
        shutil.rmtree(temp_dir)
        print(f"✅ 已清理臨時目錄: {temp_dir}")
    
    extractor.remove_hooks()
    print("✅ 測試 2 通過！\n")


def test_multiple_layers():
    """測試多層同時擷取。"""
    print("\n=== 測試 3: 多層同時擷取 ===")
    
    # 建立模型
    model = Simple3DCNN_InstanceNorm(in_channels=1, num_classes=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # 測試所有 4 個 block
    target_layers = ['block1', 'block2', 'block3', 'block4']
    extractor = ActivationExtractor(model, target_layers, device)
    extractor.register_hooks()
    
    # 建立測試輸入
    test_input = torch.randn(1, 1, 128, 128, 128).to(device)
    
    # 擷取資料
    results = extractor.extract(test_input, target_class=1, subject_id="multi_layer_test")
    
    # 驗證所有層都被擷取
    assert len(results) == 4, f"應該擷取 4 個層，但得到 {len(results)} 個"
    
    print("✅ 成功擷取所有 4 個層:")
    for layer_name in target_layers:
        assert layer_name in results
        act_shape = results[layer_name]['activation'].shape
        grad_shape = results[layer_name]['gradient'].shape
        print(f"   {layer_name}: activation {act_shape}, gradient {grad_shape}")
    
    extractor.remove_hooks()
    print("✅ 測試 3 通過！\n")


def test_inference_correctness():
    """測試擷取過程不影響模型推論結果。"""
    print("\n=== 測試 4: 推論正確性 ===")
    
    # 建立模型
    model = Simple3DCNN_InstanceNorm(in_channels=1, num_classes=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # 建立測試輸入
    test_input = torch.randn(1, 1, 128, 128, 128).to(device)
    
    # 不使用 extractor 的推論結果
    with torch.no_grad():
        output_without_hooks = model(test_input)
    
    # 使用 extractor 的推論結果
    target_layers = ['block4']
    extractor = ActivationExtractor(model, target_layers, device)
    extractor.register_hooks()
    
    with torch.no_grad():
        output_with_hooks = model(test_input)
    
    # 驗證結果相同
    assert torch.allclose(output_without_hooks, output_with_hooks, atol=1e-6), \
        "使用 hooks 後推論結果改變"
    
    print("✅ 推論結果驗證:")
    print(f"   不使用 hooks: {output_without_hooks}")
    print(f"   使用 hooks: {output_with_hooks}")
    print("✅ 推論結果一致！")
    
    extractor.remove_hooks()
    print("✅ 測試 4 通過！\n")


def main():
    """執行所有測試。"""
    print("=" * 60)
    print("開始測試 ActivationExtractor")
    print("=" * 60)
    
    try:
        test_activation_extractor_basic()
        test_save_and_load()
        test_multiple_layers()
        test_inference_correctness()
        
        print("=" * 60)
        print("✅ 所有測試通過！")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ 測試失敗: {e}")
        return False
    except Exception as e:
        print(f"\n❌ 發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
