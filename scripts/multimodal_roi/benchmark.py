"""
Performance Benchmark Script
效能基準測試腳本

測試不同配置下的訓練速度和 VRAM 使用
"""

import torch
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import DEVICE
from resnet3d_mini import MultiModalFeatureExtractor


def benchmark_batch_sizes():
    """測試不同 batch size 的效能"""
    print("="*80)
    print("Batch Size Benchmark")
    print("="*80)
    
    model = MultiModalFeatureExtractor(feature_dim=64, initial_filters=32).to(DEVICE)
    model.eval()
    
    # 測試不同的 batch size
    batch_sizes = [2, 4, 8, 12, 16, 20, 24, 28, 32]
    results = []
    
    print(f"\nDevice: {DEVICE}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("\nTesting different batch sizes...")
    print("-"*80)
    
    for bs in batch_sizes:
        try:
            # 清理 GPU 緩存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            # 創建測試數據
            t1 = torch.randn(bs, 116, 1, 32, 32, 32).to(DEVICE)
            t2 = torch.randn(bs, 116, 1, 32, 32, 32).to(DEVICE)
            dwi = torch.randn(bs, 116, 1, 32, 32, 32).to(DEVICE)
            
            # 預熱
            with torch.no_grad():
                _ = model(t1, t2, dwi)
            
            # 同步 GPU
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # 測試
            start = time.time()
            num_iterations = 10
            
            with torch.no_grad():
                for _ in range(num_iterations):
                    _ = model(t1, t2, dwi)
            
            # 同步 GPU
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            elapsed = time.time() - start
            
            # 統計
            if torch.cuda.is_available():
                vram_allocated = torch.cuda.max_memory_allocated() / 1e9
                vram_reserved = torch.cuda.max_memory_reserved() / 1e9
            else:
                vram_allocated = 0
                vram_reserved = 0
            
            throughput = (bs * num_iterations) / elapsed
            time_per_sample = elapsed / (bs * num_iterations)
            
            result = {
                'batch_size': bs,
                'time': elapsed,
                'throughput': throughput,
                'time_per_sample': time_per_sample,
                'vram_allocated': vram_allocated,
                'vram_reserved': vram_reserved,
            }
            results.append(result)
            
            print(f"Batch Size: {bs:3d} | "
                  f"Time: {elapsed:6.2f}s | "
                  f"Throughput: {throughput:6.1f} samples/s | "
                  f"VRAM: {vram_allocated:5.2f} GB (allocated) / {vram_reserved:5.2f} GB (reserved)")
            
            # 清理
            del t1, t2, dwi
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Batch Size: {bs:3d} | [FAIL] Out of Memory")
                break
            else:
                print(f"Batch Size: {bs:3d} | [FAIL] {e}")
                break
    
    # 分析結果
    print("\n" + "="*80)
    print("Analysis")
    print("="*80)
    
    if len(results) > 0:
        # 找到最佳配置
        best_throughput = max(results, key=lambda x: x['throughput'])
        best_efficiency = max(results, key=lambda x: x['throughput'] / x['vram_allocated'] if x['vram_allocated'] > 0 else 0)
        
        print(f"\n[OK] Tested {len(results)} configurations")
        print(f"\nBest throughput:")
        print(f"  Batch Size: {best_throughput['batch_size']}")
        print(f"  Throughput: {best_throughput['throughput']:.1f} samples/s")
        print(f"  VRAM: {best_throughput['vram_allocated']:.2f} GB")
        
        print(f"\nBest efficiency (throughput/VRAM):")
        print(f"  Batch Size: {best_efficiency['batch_size']}")
        print(f"  Throughput: {best_efficiency['throughput']:.1f} samples/s")
        print(f"  VRAM: {best_efficiency['vram_allocated']:.2f} GB")
        print(f"  Efficiency: {best_efficiency['throughput'] / best_efficiency['vram_allocated']:.1f} samples/s/GB")
        
        # 推薦配置
        print(f"\n[RECOMMEND] Recommended configurations:")
        
        # 保守配置 (使用 50% VRAM)
        if torch.cuda.is_available():
            total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            target_vram_conservative = total_vram * 0.5
            target_vram_aggressive = total_vram * 0.8
            
            conservative = min(results, key=lambda x: abs(x['vram_allocated'] - target_vram_conservative))
            aggressive = min(results, key=lambda x: abs(x['vram_allocated'] - target_vram_aggressive))
            
            print(f"\n  Conservative (50% VRAM):")
            print(f"    BATCH_SIZE = {conservative['batch_size']}")
            print(f"    Expected VRAM: {conservative['vram_allocated']:.2f} GB / {total_vram:.2f} GB")
            print(f"    Expected throughput: {conservative['throughput']:.1f} samples/s")
            
            print(f"\n  Aggressive (80% VRAM):")
            print(f"    BATCH_SIZE = {aggressive['batch_size']}")
            print(f"    Expected VRAM: {aggressive['vram_allocated']:.2f} GB / {total_vram:.2f} GB")
            print(f"    Expected throughput: {aggressive['throughput']:.1f} samples/s")
        
        # 估算訓練時間
        print(f"\n[INFO] Training time estimation (for 500 samples, 100 epochs):")
        for config_name, config in [('Conservative', conservative), ('Aggressive', aggressive)]:
            samples_per_epoch = 500
            num_epochs = 100
            total_samples = samples_per_epoch * num_epochs
            estimated_time = total_samples / config['throughput']
            
            print(f"  {config_name}: {estimated_time/3600:.1f} hours ({estimated_time/60:.0f} minutes)")
    
    else:
        print("[FAIL] No successful configurations")
    
    return results


def benchmark_model_sizes():
    """測試不同模型大小的效能"""
    print("\n" + "="*80)
    print("Model Size Benchmark")
    print("="*80)
    
    configs = [
        {"name": "Small", "initial_filters": 16, "feature_dim": 32},
        {"name": "Medium", "initial_filters": 32, "feature_dim": 64},
        {"name": "Large", "initial_filters": 64, "feature_dim": 128},
    ]
    
    batch_size = 8  # 固定 batch size
    
    print(f"\nFixed batch size: {batch_size}")
    print("-"*80)
    
    for config in configs:
        try:
            # 清理 GPU 緩存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            # 創建模型
            model = MultiModalFeatureExtractor(
                feature_dim=config['feature_dim'],
                initial_filters=config['initial_filters']
            ).to(DEVICE)
            model.eval()
            
            # 計算參數量
            total_params = sum(p.numel() for p in model.parameters())
            
            # 創建測試數據
            t1 = torch.randn(batch_size, 116, 1, 32, 32, 32).to(DEVICE)
            t2 = torch.randn(batch_size, 116, 1, 32, 32, 32).to(DEVICE)
            dwi = torch.randn(batch_size, 116, 1, 32, 32, 32).to(DEVICE)
            
            # 預熱
            with torch.no_grad():
                _ = model(t1, t2, dwi)
            
            # 測試
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start = time.time()
            num_iterations = 10
            
            with torch.no_grad():
                for _ in range(num_iterations):
                    _ = model(t1, t2, dwi)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            elapsed = time.time() - start
            
            # 統計
            if torch.cuda.is_available():
                vram = torch.cuda.max_memory_allocated() / 1e9
            else:
                vram = 0
            
            throughput = (batch_size * num_iterations) / elapsed
            
            print(f"{config['name']:8s} | "
                  f"Params: {total_params/1e6:5.2f}M | "
                  f"Time: {elapsed:6.2f}s | "
                  f"Throughput: {throughput:6.1f} samples/s | "
                  f"VRAM: {vram:5.2f} GB")
            
            # 清理
            del model, t1, t2, dwi
            
        except Exception as e:
            print(f"{config['name']:8s} | [FAIL] {e}")


def main():
    """主函數"""
    print("="*80)
    print("Multi-modal ROI Pipeline - Performance Benchmark")
    print("="*80)
    
    # 系統信息
    print(f"\nSystem Information:")
    print(f"  Device: {DEVICE}")
    
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"  CUDA Version: {torch.version.cuda}")
    
    print(f"  PyTorch Version: {torch.__version__}")
    
    # 運行基準測試
    print("\n" + "="*80)
    print("Starting Benchmarks...")
    print("="*80)
    
    # 測試 1: Batch Size
    results = benchmark_batch_sizes()
    
    # 測試 2: Model Size
    if torch.cuda.is_available():
        benchmark_model_sizes()
    
    print("\n" + "="*80)
    print("Benchmark Complete!")
    print("="*80)
    
    print("\nNext steps:")
    print("1. Update config.py with recommended BATCH_SIZE")
    print("2. Run training: python scripts/multimodal_roi/train.py")
    print("3. Monitor VRAM usage: nvidia-smi -l 1")


if __name__ == "__main__":
    main()
