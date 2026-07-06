import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radiology_checker.core.parser import FindingParser
from radiology_checker.core.rule_engine import RuleEngine

parser = FindingParser()
rule_engine = RuleEngine()

findings_text = "右侧胸腔可见少量积液。"
conclusion_text = "左侧胸腔积液。"

findings_parsed = parser.parse(findings_text)
conclusion_parsed = parser.parse(conclusion_text)

print("所见解析:")
for f in findings_parsed:
    print(f"  site={f.anatomical_site}, lesion={f.lesion_type}, polarity={f.polarity}, laterality={f.laterality}")

print("\n结论解析:")
for c in conclusion_parsed:
    print(f"  site={c.anatomical_site}, lesion={c.lesion_type}, polarity={c.polarity}, laterality={c.laterality}")

print("\n质量评估:")
for f in findings_parsed:
    for c in conclusion_parsed:
        quality = rule_engine._evaluate_match_quality(f, c)
        print(f"  total_score={quality.total_score}, is_match={quality.is_match}")
        print(f"  site_match={quality.site_match}, lesion_match={quality.lesion_match}, lat_match={quality.laterality_match}, polarity_match={quality.polarity_match}")

print("\n侧位冲突检测:")
for f in findings_parsed:
    for c in conclusion_parsed:
        quality = rule_engine._evaluate_match_quality(f, c)
        conflict = rule_engine._detect_laterality_conflict(f, c, findings_parsed, conclusion_parsed, quality)
        if conflict:
            print(f"  检测到侧位冲突: {conflict.explanation}")
        else:
            print(f"  未检测到侧位冲突")

print("\n直接侧位冲突检测(不依赖quality):")
for f in findings_parsed:
    for c in conclusion_parsed:
        conflict = rule_engine._detect_laterality_conflict(f, c, findings_parsed, conclusion_parsed)
        if conflict:
            print(f"  检测到侧位冲突: {conflict.explanation}")
        else:
            print(f"  未检测到侧位冲突")

print("\n_is_laterality_conflict测试:")
result = rule_engine._is_laterality_conflict("右", "左")
print(f"  _is_laterality_conflict('右', '左') = {result}")