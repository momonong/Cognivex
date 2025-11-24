#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試 HuggingFace 整合

這個腳本測試 CDDA 系統是否能正確使用 HuggingFace 模型
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_huggingface_availability():
    """測試 HuggingFace 是否可用"""
    print("\n" + "="*80)
    print("測試 1: HuggingFace 可用性")
    print("="*80)
    
    try:
        from app.services.llm_providers import huggingface
        
        if huggingface.check_availability():
            print("✅ HuggingFace (transformers) 已安裝")
            return True
        else:
            print("❌ HuggingFace (transformers) 未安裝")
            print("請執行: pip install transformers torch accelerate")
            return False
    except Exception as e:
        print(f"❌ 錯誤: {e}")
        return False


def test_model_path_detection():
    """測試模型路徑偵測"""
    print("\n" + "="*80)
    print("測試 2: 模型路徑偵測")
    print("="*80)
    
    try:
        from app.services.llm_providers import huggingface
        
        # 測試路徑
        test_paths = [
            "D:/hf_models/gpt-oss-20b",
            "D:/hf_models/medgemma-27b",
            "D:/hf_models/phi-3-mini",
        ]
        
        found_models = []
        
        for path in test_paths:
            info = huggingface.get_model_info(path)
            if info['exists']:
                print(f"✅ 找到模型: {path}")
                print(f"   SafeTensors 檔案: {info['safetensors_count']}")
                found_models.append(path)
            else:
                print(f"❌ 找不到模型: {path}")
        
        if found_models:
            print(f"\n找到 {len(found_models)} 個模型")
            return True
        else:
            print("\n❌ 沒有找到任何模型")
            print("請下載模型到以下目錄之一:")
            for path in test_paths:
                print(f"  - {path}")
            return False
            
    except Exception as e:
        print(f"❌ 錯誤: {e}")
        return False


def test_cdda_agent_initialization():
    """測試 CDDA Agent 初始化"""
    print("\n" + "="*80)
    print("測試 3: CDDA Agent 初始化 (規則模式)")
    print("="*80)
    
    try:
        from app.agents.cdda_agent import CDDAAgent
        
        # 測試規則模式 (不需要 LLM)
        agent = CDDAAgent(
            use_llm=False,
            verbose=True
        )
        
        print("✅ CDDA Agent 初始化成功 (規則模式)")
        return True
        
    except Exception as e:
        print(f"❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cdda_agent_with_huggingface():
    """測試 CDDA Agent 使用 HuggingFace"""
    print("\n" + "="*80)
    print("測試 4: CDDA Agent 初始化 (HuggingFace 模式)")
    print("="*80)
    
    try:
        from app.agents.cdda_agent import CDDAAgent
        from app.services.llm_providers import huggingface
        
        # 尋找可用的模型
        test_paths = [
            "D:/hf_models/phi-3-mini",
            "D:/hf_models/gpt-oss-20b",
            "D:/hf_models/medgemma-27b",
        ]
        
        model_path = None
        for path in test_paths:
            info = huggingface.get_model_info(path)
            if info['exists']:
                model_path = path
                break
        
        if not model_path:
            print("⚠️  跳過: 找不到可用的模型")
            print("請先下載模型以測試 HuggingFace 整合")
            return None
        
        print(f"使用模型: {model_path}")
        
        # 測試 HuggingFace 模式
        agent = CDDAAgent(
            orchestrator_model="test-model",
            orchestrator_model_path=model_path,
            consultant_model="test-model",
            consultant_model_path=model_path,
            use_llm=True,
            load_in_8bit=True,
            verbose=True
        )
        
        print("✅ CDDA Agent 初始化成功 (HuggingFace 模式)")
        return True
        
    except Exception as e:
        print(f"❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_file_search():
    """測試檔案搜尋邏輯"""
    print("\n" + "="*80)
    print("測試 5: NIfTI 檔案搜尋")
    print("="*80)
    
    try:
        # 測試受試者
        test_subjects = [
            ("sub-0001", "AD"),
            ("sub-0015", "NC"),
            ("sub-0005", "MCI"),
        ]
        
        found_files = []
        
        for subject_id, label in test_subjects:
            # 可能的路徑
            possible_paths = [
                f"data/MRI_processed/{label}/{subject_id}/anat/{subject_id}_T1w.nii.gz",
                f"data/MRI_processed/{label}/{subject_id}/{subject_id}_T1w.nii.gz",
                f"data/sMRI/{label}/{subject_id}/anat/{subject_id}_T1w.nii.gz",
                f"data/fMRI/{label}/{subject_id}/func/{subject_id}_task-rest_bold.nii.gz",
            ]
            
            found = False
            for path in possible_paths:
                if Path(path).exists():
                    print(f"✅ 找到: {subject_id} -> {path}")
                    found_files.append(path)
                    found = True
                    break
            
            if not found:
                print(f"❌ 找不到: {subject_id} ({label})")
        
        if found_files:
            print(f"\n找到 {len(found_files)} 個檔案")
            return True
        else:
            print("\n❌ 沒有找到任何檔案")
            print("請確認資料位於 data/MRI_processed/ 目錄")
            return False
            
    except Exception as e:
        print(f"❌ 錯誤: {e}")
        return False


def main():
    """執行所有測試"""
    print("\n" + "="*80)
    print("CDDA HuggingFace 整合測試")
    print("="*80)
    
    results = []
    
    # 測試 1: HuggingFace 可用性
    results.append(("HuggingFace 可用性", test_huggingface_availability()))
    
    # 測試 2: 模型路徑偵測
    results.append(("模型路徑偵測", test_model_path_detection()))
    
    # 測試 3: CDDA Agent 初始化 (規則模式)
    results.append(("CDDA Agent (規則模式)", test_cdda_agent_initialization()))
    
    # 測試 4: CDDA Agent 使用 HuggingFace
    result = test_cdda_agent_with_huggingface()
    if result is not None:
        results.append(("CDDA Agent (HuggingFace)", result))
    
    # 測試 5: 檔案搜尋
    results.append(("NIfTI 檔案搜尋", test_file_search()))
    
    # 總結
    print("\n" + "="*80)
    print("測試總結")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通過" if result else "❌ 失敗"
        print(f"{status}: {name}")
    
    print(f"\n總計: {passed}/{total} 測試通過")
    
    if passed == total:
        print("\n🎉 所有測試通過！系統已準備就緒。")
        return 0
    else:
        print("\n⚠️  部分測試失敗，請檢查上述錯誤訊息。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
