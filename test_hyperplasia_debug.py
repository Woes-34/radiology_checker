import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radiology_checker.core.parser import FindingParser
from radiology_checker.core.rule_engine import RuleEngine

parser = FindingParser()
rule_engine = RuleEngine()

findings_text = "腰椎椎体边缘见骨质增生。"
conclusion_text = "腰椎未见骨质增生。"

findings_parsed = parser.parse(findings_text)
conclusion_parsed = parser.parse(conclusion_text)

print("所见解析:")
for f in findings_parsed:
    print(f"  site={f.anatomical_site}, lesion={f.lesion_type}, polarity={f.polarity}, raw={f.raw_text}")

print("\n结论解析:")
for c in conclusion_parsed:
    print(f"  site={c.anatomical_site}, lesion={c.lesion_type}, polarity={c.polarity}, raw={c.raw_text}")

print("\n质量评估:")
for f in findings_parsed:
    for c in conclusion_parsed:
        quality = rule_engine._evaluate_match_quality(f, c)
        print(f"  total_score={quality.total_score}, is_match={quality.is_match}")
        print(f"  details={quality.details}")

print("\n显式否定冲突检测:")
for f in findings_parsed:
    for c in conclusion_parsed:
        conflict = rule_engine._detect_explicit_negation_conflict(f, c)
        if conflict:
            print(f"  检测到: {conflict.explanation}")
        else:
            print(f"  未检测到")

print("\n极性冲突检测:")
for f in findings_parsed:
    for c in conclusion_parsed:
        quality = rule_engine._evaluate_match_quality(f, c)
        conflict = rule_engine._detect_polarity_conflict(f, c, quality)
        if conflict:
            print(f"  检测到: {conflict.explanation}")
        else:
            print(f"  未检测到")

print("\n_is_site_related测试:")
result = rule_engine._is_site_related("椎体", "腰椎")
print(f"  _is_site_related('椎体', '腰椎') = {result}")