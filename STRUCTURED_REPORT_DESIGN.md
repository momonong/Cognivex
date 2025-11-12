# 🎯 Structured Clinical Report Design

## 概念

將長篇文字報告改為結構化數據，讓 Dashboard 更直觀、更豐富、更易於快速理解。

## Structured Output Schema

### 1. Risk Assessment (風險評估)
```json
{
    "risk_level": "High Risk" | "Low Risk" | "Moderate Risk",
    "confidence_score": 0.85,
    "primary_finding": "Brief primary finding in 1-2 sentences"
}
```

### 2. Key Findings (關鍵發現)
```json
{
    "structural_changes": [
        {
            "finding": "Bilateral hippocampal atrophy",
            "severity": "Moderate" | "Mild" | "Severe",
            "significance": "High" | "Medium" | "Low"
        }
    ],
    "volumetric_analysis": [
        {
            "region": "Temporal Lobe",
            "change": "Reduced volume",
            "percentage": "-15%"
        }
    ]
}
```

### 3. Clinical Interpretation (臨床解釋)
```json
{
    "summary": "2-3 sentence summary",
    "ad_indicators": [
        "Hippocampal atrophy",
        "Temporal lobe volume reduction",
        "Amygdala changes"
    ],
    "protective_factors": [
        "Preserved frontal lobe function",
        "Normal white matter integrity"
    ]
}
```

### 4. Recommendations (建議)
```json
{
    "immediate_actions": [
        "Follow-up MRI in 6 months",
        "Cognitive assessment recommended"
    ],
    "monitoring": [
        "Track memory function",
        "Monitor daily activities"
    ],
    "additional_tests": [
        "PET scan consideration",
        "CSF biomarkers"
    ]
}
```

## Dashboard 視覺化設計

### Layout Structure

```
┌─────────────────────────────────────────────────────────┐
│ Language: [中文 ▼]                                       │
│                                                          │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ 阿茲海默症風險評估報告                                │ │
│ │ Subject: sub_0001 | Time: 2024-11-12 15:30          │ │
│ └─────────────────────────────────────────────────────┘ │
│                                                          │
│ ┌──────────┬──────────┬──────────┬──────────┐          │
│ │ 臨床診斷  │ AI 預測  │ 信心度   │ 模型     │          │
│ │ NC       │ NC       │ 85%      │ RF       │          │
│ └──────────┴──────────┴──────────┴──────────┘          │
│                                                          │
│ ✓ 診斷一致                                               │
│                                                          │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ 主要發現 Primary Finding                             │ │
│ │ ─────────────────────────────────────────────────── │ │
│ │ 雙側海馬迴體積輕度減少，與早期阿茲海默症病理變化一致。│ │
│ │ 建議進行追蹤評估以監測病程進展。                      │ │
│ └─────────────────────────────────────────────────────┘ │
│                                                          │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ 關鍵發現 Key Findings                                │ │
│ │ ─────────────────────────────────────────────────── │ │
│ │ 結構性變化                                           │ │
│ │ • 雙側海馬迴萎縮 (中度) ⚠️                           │ │
│ │ • 顳葉體積減少 (輕度) ⚠️                             │ │
│ │                                                      │ │
│ │ 體積分析                                             │ │
│ │ • 海馬迴: -12% ▼                                     │ │
│ │ • 杏仁核: -8% ▼                                      │ │
│ │ • 額葉: 正常 ✓                                       │ │
│ └─────────────────────────────────────────────────────┘ │
│                                                          │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ 臨床解釋 Clinical Interpretation                     │ │
│ │ ─────────────────────────────────────────────────── │ │
│ │ AD 指標                                              │ │
│ │ ⚠️ 海馬迴萎縮                                         │ │
│ │ ⚠️ 顳葉體積減少                                       │ │
│ │ ⚠️ 杏仁核變化                                         │ │
│ │                                                      │ │
│ │ 保護因子                                             │ │
│ │ ✓ 額葉功能保留                                       │ │
│ │ ✓ 白質完整性正常                                     │ │
│ └─────────────────────────────────────────────────────┘ │
│                                                          │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ 建議 Recommendations                                 │ │
│ │ ─────────────────────────────────────────────────── │ │
│ │ 立即行動                                             │ │
│ │ 1. 6個月後追蹤 MRI                                   │ │
│ │ 2. 建議認知功能評估                                  │ │
│ │                                                      │ │
│ │ 監測項目                                             │ │
│ │ • 記憶功能追蹤                                       │ │
│ │ • 日常活動監測                                       │ │
│ │                                                      │ │
│ │ 額外檢查                                             │ │
│ │ • 考慮 PET 掃描                                      │ │
│ │ • CSF 生物標記                                       │ │
│ └─────────────────────────────────────────────────────┘ │
│                                                          │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ 重要腦區分析 Key Brain Regions                       │ │
│ │ ─────────────────────────────────────────────────── │ │
│ │ [表格: 排名 | 名稱 | 重要性 | 分類 | 半球]            │ │
│ └─────────────────────────────────────────────────────┘ │
│                                                          │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ 醫師備註 Physician's Notes                           │ │
│ │ ─────────────────────────────────────────────────── │ │
│ │ [文字輸入框]                                         │ │
│ └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## 優勢

### 1. 更直觀
- ✅ 視覺化的嚴重程度指標
- ✅ 清晰的分類結構
- ✅ 快速掃描關鍵資訊

### 2. 更豐富
- ✅ 多維度資訊呈現
- ✅ 量化指標（百分比、嚴重度）
- ✅ 視覺化元素（圖示、顏色）

### 3. 更簡潔
- ✅ 結構化數據易於解析
- ✅ 避免冗長文字
- ✅ 重點突出

### 4. 更專業
- ✅ 標準化報告格式
- ✅ 可量化的指標
- ✅ 易於比較和追蹤

## 實現方案

### Phase 1: 修改 LLM Prompt
```python
synthesis_prompt = f"""
Generate a structured clinical report in JSON format:

{{
    "risk_assessment": {{
        "level": "High Risk" | "Low Risk",
        "confidence": {confidence},
        "primary_finding": "1-2 sentence summary"
    }},
    "key_findings": {{
        "structural_changes": [
            {{
                "finding": "description",
                "severity": "Mild|Moderate|Severe",
                "significance": "High|Medium|Low"
            }}
        ],
        "volumetric_analysis": [
            {{
                "region": "region name",
                "change": "description",
                "percentage": "percentage change"
            }}
        ]
    }},
    "clinical_interpretation": {{
        "summary": "2-3 sentence summary",
        "ad_indicators": ["indicator1", "indicator2"],
        "protective_factors": ["factor1", "factor2"]
    }},
    "recommendations": {{
        "immediate_actions": ["action1", "action2"],
        "monitoring": ["item1", "item2"],
        "additional_tests": ["test1", "test2"]
    }}
}}

Subject: {subject_id}
Classification: {classification}
Confidence: {confidence}
Key Regions: {top_regions_text}
"""
```

### Phase 2: 解析 JSON Response
```python
import json

# Parse LLM response
structured_report = json.loads(report_en)

# Store in state
return {
    "structured_report": structured_report,
    "trace_log": state.get("trace_log", []) + [trace]
}
```

### Phase 3: 視覺化渲染
```python
def render_structured_report(structured_report, lang="中文"):
    # Primary Finding
    st.markdown("### 主要發現" if lang == "中文" else "### Primary Finding")
    st.info(structured_report["risk_assessment"]["primary_finding"])
    
    # Key Findings
    st.markdown("### 關鍵發現" if lang == "中文" else "### Key Findings")
    
    for change in structured_report["key_findings"]["structural_changes"]:
        severity_icon = {
            "Severe": "🔴",
            "Moderate": "🟡",
            "Mild": "🟢"
        }[change["severity"]]
        
        st.markdown(f"{severity_icon} {change['finding']} ({change['severity']})")
    
    # Clinical Interpretation
    st.markdown("### 臨床解釋" if lang == "中文" else "### Clinical Interpretation")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**AD 指標**")
        for indicator in structured_report["clinical_interpretation"]["ad_indicators"]:
            st.markdown(f"⚠️ {indicator}")
    
    with col2:
        st.markdown("**保護因子**")
        for factor in structured_report["clinical_interpretation"]["protective_factors"]:
            st.markdown(f"✓ {factor}")
    
    # Recommendations
    st.markdown("### 建議" if lang == "中文" else "### Recommendations")
    
    for action in structured_report["recommendations"]["immediate_actions"]:
        st.markdown(f"1. {action}")
```

## 時間估計

- **Phase 1** (修改 Prompt): 30 分鐘
- **Phase 2** (解析 JSON): 20 分鐘
- **Phase 3** (視覺化渲染): 1-2 小時
- **測試與調整**: 1 小時

**總計**: 約 3-4 小時

## 建議

我建議實施這個 structured output 方案，因為：

1. ✅ **更符合臨床需求** - 醫師需要快速掃描關鍵資訊
2. ✅ **更易於維護** - 結構化數據易於更新和擴展
3. ✅ **更好的用戶體驗** - 視覺化呈現更直觀
4. ✅ **可追蹤性** - 結構化數據易於比較不同時間點的報告

你想要我現在開始實施這個方案嗎？

---

*設計日期: 2024年*
*設計理念: 結構化、視覺化、專業化*
