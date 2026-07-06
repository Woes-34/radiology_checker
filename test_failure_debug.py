import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radiology_checker.core.parser import FindingParser
from radiology_checker.core.rule_engine import RuleEngine

parser = FindingParser()
rule_engine = RuleEngine()

test_cases = [
    {
        "name": "上叶vs下叶",
        "findings": "右肺上叶见结节影。",
        "conclusion": "右肺下叶结节。"
    },
    {
        "name": "左肾vs右肾",
        "findings": "左肾见一约1.0cm囊肿。",
        "conclusion": "右肾囊肿。"
    },
    {
        "name": "肺纹理增多",
        "findings": "双肺纹理增多、紊乱，未见明显异常密度影。",
        "conclusion": "双肺纹理增多，未见明显异常。"
    },
    {
        "name": "脑梗死vs无梗死",
        "findings": "右侧基底节区见片状低密度影，边界不清，考虑脑梗死。",
        "conclusion": "头颅CT未见脑梗死。"
    },
    {
        "name": "出血vs无出血",
        "findings": "左侧丘脑见高密度影，考虑出血。",
        "conclusion": "颅内未见出血。"
    }
]

for case in test_cases:
    print(f"\n{'='*60}")
    print(f"案例: {case['name']}")
    print(f"所见: {case['findings']}")
    print(f"结论: {case['conclusion']}")
    
    findings_parsed = parser.parse(case['findings'])
    conclusion_parsed = parser.parse(case['conclusion'])
    
    print("\n所见解析:")
    for f in findings_parsed:
        print(f"  site={f.anatomical_site}, lesion={f.lesion_type}, polarity={f.polarity}, lat={f.laterality}, raw={f.raw_text}")
    
    print("\n结论解析:")
    for c in conclusion_parsed:
        print(f"  site={c.anatomical_site}, lesion={c.lesion_type}, polarity={c.polarity}, lat={c.laterality}, raw={c.raw_text}")
    
    print("\n质量评估:")
    for f in findings_parsed:
        for c in conclusion_parsed:
            quality = rule_engine._evaluate_match_quality(f, c)
            print(f"  total_score={quality.total_score}, is_match={quality.is_match}")
            print(f"  details={quality.details}")
    
    conflicts, conf_level = rule_engine.check_conflicts(findings_parsed, conclusion_parsed)
    print(f"\n规则引擎检测: {len(conflicts)} 个冲突")
    for c in conflicts:
        print(f"  - {c.explanation} (置信度: {c.confidence})")
    
    print(f"置信度等级: {conf_level}")