import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radiology_checker.core.parser import FindingParser

test_cases = [
    {
        "name": "侧位矛盾",
        "findings": "右侧胸腔可见少量积液。",
        "conclusion": "左侧胸腔积液。",
    },
    {
        "name": "增生矛盾",
        "findings": "腰椎椎体边缘见骨质增生。",
        "conclusion": "腰椎未见骨质增生。",
    },
    {
        "name": "结节矛盾",
        "findings": "左肺上叶见一约0.5cm结节影，边界清晰，密度均匀。",
        "conclusion": "双肺未见异常。",
    },
]

parser = FindingParser()

for case in test_cases:
    print(f"\n=== {case['name']} ===")
    print(f"\n所见: {case['findings']}")
    findings_parsed = parser.parse(case['findings'])
    for i, f in enumerate(findings_parsed, 1):
        print(f"  Finding {i}: site={f.anatomical_site}, lesion={f.lesion_type}, polarity={f.polarity}, laterality={f.laterality}, raw={f.raw_text}")
    
    print(f"\n结论: {case['conclusion']}")
    conclusion_parsed = parser.parse(case['conclusion'])
    for i, c in enumerate(conclusion_parsed, 1):
        print(f"  Conclusion {i}: site={c.anatomical_site}, lesion={c.lesion_type}, polarity={c.polarity}, laterality={c.laterality}, raw={c.raw_text}")