# examples/activation_extractor_example.py

"""
ActivationExtractor 使用範例

展示如何使用 ActivationExtractor 從 3D CNN 模型擷取 activation 和 gradient。
"""

import torch
import os
import sys

# 加入專案根目錄到路徑
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from app.core.xai.activation_extractor import ActivationExtractor
from app.core.cnn_3d.model import Simple3DCNN_InstanceNorm


def example_basic_usage():
    """基本使用範例。"""
    print("\n=== 範例 1: 基本使用 ===\n")
    
    # 1. 建立模型
    model = Simple3DCNN_InstanceNorm(in_channels=1, num_classes=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")
    
    # 2. 建立 ActivationExtractor
    target_layers = ['block4', 'block3']  # 指定要擷取的層
    extractor = ActivationExtractor(model, target_layers, device)
    
    # 3. 註冊 hooks
    extractor.register_hooks()
    
    # 4. 準備輸入資料 (這裡使用隨機資料作為示範)
    # 實際使用時，這應該是經過預處理的 NIfTI 影像
    test_input = torch.randn(1, 1, 128, 128, 128).to(device)
    
    # 5. 擷取 activations 和 gradients
    results = extractor.extract(
        input_tensor=test_input,
        target_class=1,  # 1 = AD, 0 = NC
        subject_id="example_subject"
    )
    
    # 6. 查看結果
    for layer_name, data in results.items():
        print(f"\n層: {layer_name}")
        print(f"  Activation shape: {data['activation'].shape}")
        print(f"  Gradient shape: {data['gradient'].shape}")
        print(f"  Target score: {data['metadata']['target_score']:.4f}")
    
    # 7. 移除 hooks (可選，解構時會自動移除)
    extractor.remove_hooks()
    
    print("\n✅ 範例 1 完成！")


def example_save_and_load():
    """儲存和載入範例。"""
    print("\n=== 範例 2: 儲存和載入 ===\n")
    
    # 建立模型和 extractor
    model = Simple3DCNN_InstanceNorm(in_channels=1, num_classes=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    target_layers = ['block4']
    extractor = ActivationExtractor(model, target_layers, device)
    extractor.register_hooks()
    
    # 擷取資料
    test_input = torch.randn(1, 1, 128, 128, 128).to(device)
    results = extractor.extract(test_input, target_class=0, subject_id="sub-001")
    
    # 儲存到磁碟
    output_dir = os.path.join(PROJECT_ROOT, "output/activations")
    extractor.save_to_disk(results, output_dir, subject_id="sub-001")
    print(f"\n✅ 資料已儲存到: {output_dir}")
    
    # 載入資料
    saved_file = os.path.join(output_dir, "sub-001_block4_activations.pt")
    loaded_data = extractor.load_from_disk(saved_file)
    
    print(f"\n✅ 資料已載入:")
    print(f"  Activation shape: {loaded_data['activation'].shape}")
    print(f"  Subject ID: {loaded_data['metadata']['subject_id']}")
    print(f"  Timestamp: {loaded_data['metadata']['timestamp']}")
    
    extractor.remove_hooks()
    print("\n✅ 範例 2 完成！")


def example_with_real_model():
    """使用真實模型權重的範例。"""
    print("\n=== 範例 3: 使用真實模型權重 ===\n")
    
    # 檢查模型權重是否存在
    model_path = os.path.join(PROJECT_ROOT, "model/cnn_3d/cnn_3d_fold_1.pth")
    
    if not os.path.exists(model_path):
        print(f"⚠️ 找不到模型權重: {model_path}")
        print("跳過此範例")
        return
    
    # 載入訓練好的模型
    model = Simple3DCNN_InstanceNorm(in_channels=1, num_classes=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"✅ 已載入模型權重: {model_path}")
    
    # 建立 extractor
    target_layers = ['block4', 'block3', 'block2', 'block1']
    extractor = ActivationExtractor(model, target_layers, device)
    extractor.register_hooks()
    
    # 使用隨機輸入 (實際使用時應該是真實的 NIfTI 資料)
    test_input = torch.randn(1, 1, 128, 128, 128).to(device)
    
    # 擷取所有層的資料
    results = extractor.extract(test_input, target_class=1, subject_id="real_model_test")
    
    print(f"\n✅ 成功擷取 {len(results)} 個層的資料:")
    for layer_name in target_layers:
        if layer_name in results:
            act_shape = results[layer_name]['activation'].shape
            print(f"  {layer_name}: {act_shape}")
    
    extractor.remove_hooks()
    print("\n✅ 範例 3 完成！")


def main():
    """執行所有範例。"""
    print("=" * 60)
    print("ActivationExtractor 使用範例")
    print("=" * 60)
    
    try:
        example_basic_usage()
        example_save_and_load()
        example_with_real_model()
        
        print("\n" + "=" * 60)
        print("✅ 所有範例執行完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 發生錯誤: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
