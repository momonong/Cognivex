"""
測試 Structured Report 生成
"""

import json

# Mock structured report for testing
mock_structured_report = {
    "en": {
        "risk_assessment": {
            "level": "Low Risk",
            "confidence": 0.53,
            "primary_finding": "Mild bilateral hippocampal volume reduction observed, consistent with early-stage Alzheimer's pathology. Follow-up assessment recommended to monitor disease progression."
        },
        "key_findings": {
            "structural_changes": [
                {
                    "finding": "Bilateral hippocampal atrophy",
                    "severity": "Moderate",
                    "significance": "High"
                },
                {
                    "finding": "Temporal lobe volume reduction",
                    "severity": "Mild",
                    "significance": "Medium"
                }
            ],
            "volumetric_analysis": [
                {
                    "region": "Hippocampus",
                    "change": "Volume reduction",
                    "percentage": "-12%"
                },
                {
                    "region": "Amygdala",
                    "change": "Volume reduction",
                    "percentage": "-8%"
                },
                {
                    "region": "Frontal Lobe",
                    "change": "Normal",
                    "percentage": "0%"
                }
            ]
        },
        "clinical_interpretation": {
            "summary": "The observed structural changes are consistent with early Alzheimer's disease pathology. The bilateral hippocampal atrophy and temporal lobe volume reduction are characteristic findings.",
            "ad_indicators": [
                "Hippocampal atrophy",
                "Temporal lobe volume reduction",
                "Amygdala changes"
            ],
            "protective_factors": [
                "Preserved frontal lobe function",
                "Normal white matter integrity"
            ]
        },
        "recommendations": {
            "immediate_actions": [
                "Follow-up MRI in 6 months",
                "Cognitive assessment recommended"
            ],
            "monitoring": [
                "Track memory function",
                "Monitor daily activities"
            ],
            "additional_tests": [
                "Consider PET scan",
                "CSF biomarkers evaluation"
            ]
        },
        "limitations": [
            "This is an assistive diagnostic tool and should not be used as the sole basis for clinical decisions",
            "Results should be interpreted in conjunction with clinical assessment and other diagnostic tests"
        ]
    },
    "zh": {
        "risk_assessment": {
            "level": "低風險",
            "confidence": 0.53,
            "primary_finding": "觀察到輕度雙側海馬迴體積減少，與早期阿茲海默症病理變化一致。建議進行追蹤評估以監測病程進展。"
        },
        "key_findings": {
            "structural_changes": [
                {
                    "finding": "雙側海馬迴萎縮",
                    "severity": "Moderate",
                    "significance": "High"
                },
                {
                    "finding": "顳葉體積減少",
                    "severity": "Mild",
                    "significance": "Medium"
                }
            ],
            "volumetric_analysis": [
                {
                    "region": "海馬迴",
                    "change": "體積減少",
                    "percentage": "-12%"
                },
                {
                    "region": "杏仁核",
                    "change": "體積減少",
                    "percentage": "-8%"
                },
                {
                    "region": "額葉",
                    "change": "正常",
                    "percentage": "0%"
                }
            ]
        },
        "clinical_interpretation": {
            "summary": "觀察到的結構性變化與早期阿茲海默症病理一致。雙側海馬迴萎縮和顳葉體積減少是特徵性發現。",
            "ad_indicators": [
                "海馬迴萎縮",
                "顳葉體積減少",
                "杏仁核變化"
            ],
            "protective_factors": [
                "額葉功能保留",
                "白質完整性正常"
            ]
        },
        "recommendations": {
            "immediate_actions": [
                "6個月後追蹤 MRI",
                "建議進行認知功能評估"
            ],
            "monitoring": [
                "追蹤記憶功能",
                "監測日常活動"
            ],
            "additional_tests": [
                "考慮 PET 掃描",
                "CSF 生物標記評估"
            ]
        },
        "limitations": [
            "這是一個輔助診斷工具，不應作為臨床決策的唯一依據",
            "結果應結合臨床評估和其他診斷測試進行解釋"
        ]
    }
}

print("="*70)
print("Structured Report Test")
print("="*70)

print("\n[TEST 1] JSON Structure Validation")
print("-" * 50)

try:
    # Validate JSON structure
    json_str = json.dumps(mock_structured_report, indent=2, ensure_ascii=False)
    parsed = json.loads(json_str)
    print("✅ JSON structure is valid")
    print(f"\nLanguages available: {list(parsed.keys())}")
    
    # Check required fields
    required_fields = [
        "risk_assessment",
        "key_findings",
        "clinical_interpretation",
        "recommendations",
        "limitations"
    ]
    
    for lang in ["en", "zh"]:
        print(f"\n{lang.upper()} Report:")
        for field in required_fields:
            if field in parsed[lang]:
                print(f"  ✅ {field}")
            else:
                print(f"  ❌ {field} missing")
    
except Exception as e:
    print(f"❌ JSON validation failed: {e}")

print("\n[TEST 2] Content Preview")
print("-" * 50)

for lang in ["en", "zh"]:
    print(f"\n{lang.upper()} Report Preview:")
    report = mock_structured_report[lang]
    
    print(f"\nPrimary Finding:")
    print(f"  {report['risk_assessment']['primary_finding'][:100]}...")
    
    print(f"\nStructural Changes: {len(report['key_findings']['structural_changes'])}")
    for change in report['key_findings']['structural_changes']:
        print(f"  • {change['finding']} ({change['severity']})")
    
    print(f"\nAD Indicators: {len(report['clinical_interpretation']['ad_indicators'])}")
    for indicator in report['clinical_interpretation']['ad_indicators']:
        print(f"  ⚠️ {indicator}")

print("\n" + "="*70)
print("✅ Structured Report Test Complete!")
print("="*70)

print("\n📝 Next Steps:")
print("1. Test with real LLM generation")
print("2. Verify UI rendering")
print("3. Test language switching")

print("\n" + "="*70)
