import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os

# --- 設定你的模型路徑 ---
PATH_ORCHESTRATOR = r"D:\hf_models\Phi-4-mini-instruct"
PATH_CONSULTANT = r"D:\hf_models\Llama3.1-Aloe-Beta-8B"

def print_gpu_status(step_name):
    print(f"\n📊 [{step_name}] VRAM Status:")
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"   Allocated: {allocated:.2f} GB (實際模型佔用)")
        print(f"   Reserved:  {reserved:.2f} GB (系統預留)")
    else:
        print("   ❌ No GPU detected!")

def load_and_inspect(model_path, model_name):
    print(f"\n🔄 正在載入 {model_name} ...")
    print(f"   路徑: {model_path}")

    # --- 關鍵：最嚴格的 4-bit 設定 ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16, # 運算時用 fp16
        bnb_4bit_use_double_quant=True,       # 二次量化 (省更多)
        bnb_4bit_quant_type="nf4"             # Normal Float 4 (LLM 專用格式)
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="cuda:0", # 強制指定 GPU 0，不準去 CPU
            # device_map="auto", # 如果上面那個報錯，再換回 auto 試試
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        print(f"   ✅ {model_name} 載入成功！")
        
        # --- 驗屍：檢查權重類型 ---
        print("   🔍 檢查模型層結構 (部分):")
        # 隨便抓一層 Linear 層來檢查
        for name, module in model.named_modules():
            if "Linear" in str(type(module)) or "4bit" in str(type(module)):
                print(f"      Layer Type: {type(module)}")
                print(f"      Weight Dtype: {module.weight.dtype}")
                if "Linear4bit" in str(type(module)):
                    print("      ✅ 確認為 4-bit 量化層 (Linear4bit)")
                else:
                    print("      ❌ 警告：這不是 4-bit 層！可能載入成 FP16 了！")
                break # 檢查一層就夠了
                
        return model
    except Exception as e:
        print(f"   ❌ 載入失敗: {str(e)}")
        return None

def main():
    print("🚀 開始 VRAM 洩漏診斷...")
    
    # 1. 清理環境
    gc.collect()
    torch.cuda.empty_cache()
    print_gpu_status("Initial State")

    # 2. 載入第一個模型 (Orchestrator)
    model_a = load_and_inspect(PATH_ORCHESTRATOR, "Orchestrator (Phi-4)")
    print_gpu_status("After Phi-4")

    # 3. 載入第二個模型 (Consultant)
    model_b = load_and_inspect(PATH_CONSULTANT, "Consultant (Aloe-8B)")
    print_gpu_status("After Aloe-8B")

    # 4. 總結
    print("\n============================================")
    if torch.cuda.memory_allocated() / 1024**3 > 14:
        print("⚠️  警告：總佔用超過 14GB，量化可能未完全生效！")
    else:
        print("✅ 成功：總佔用在預期範圍內 (應 < 12GB)")
    print("============================================")

    # 防止程式立刻結束，讓你看看 GPU
    input("\n按 Enter 鍵釋放記憶體並結束...")

if __name__ == "__main__":
    main()