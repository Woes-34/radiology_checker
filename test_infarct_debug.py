import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radiology_checker.core.parser import FindingParser
from radiology_checker.core.rule_engine import RuleEngine

parser = FindingParser()
rule_engine = RuleEngine()

findings_text = "右侧基底节区见片状低密度影，边界不清，考虑脑梗死。"
conclusion_text = "头颅CT未见脑梗死。"

findings_parsed = parser.parse(findings_text)
conclusion_parsed = parser.parse(conclusion_text)

print("所见解析:")
for f in findings_parsed:
    print(f"  site={f.anatomical_site}, lesion={f.lesion_type}, polarity={f.polarity}, raw={f.raw_text}")

print("\n结论解析:")
for c in conclusion_parsed:
    print(f"  site={c.anatomical_site}, lesion={c.lesion_type}, polarity={c.polarity}, raw={c.raw_text}")

print("\n_is_site_related测试:")
print(f"  _is_site_related('基底节', '头颅') = {rule_engine._is_site_related('基底节', '头颅')}")
print(f"  _is_site_related('基底节', '颅内') = {rule_engine._is_site_related('基底节', '颅内')}")

print("\n_is_lesion_related测试:")
print(f"  _is_lesion_related('低密度影', '梗死') = {rule_engine._is_lesion_related('低密度影', '梗死')}")
print(f"  _is_lesion_related('低密度影', '脑梗死') = {rule_engine._is_lesion_related('低密度影', '脑梗死')}")

print("\n质量评估:")
for f in findings_parsed:
    for c in conclusion_parsed:
        quality = rule_engine._evaluate_match_quality(f, c)
        print(f"  total_score={quality.total_score}, is_match={quality.is_match}")
        print(f"  site_match={quality.site_match}, lesion_match={quality.lesion_match}")
        print(f"  details={quality.details}")

print("\n显式否定冲突检测:")
for f in findings_parsed:
    for c in conclusion_parsed:
        conflict = rule_engine._detect_explicit_negation_conflict(f, c)
        if conflict:
            print(f"  检测到: {conflict.explanation}")
        else:
            print(f"  未检测到")

print("\n原始文本分析:")
print(f"  所见包含'考虑': {'考虑' in findings_text}")
print(f"  结论包含'未见': {'未见' in conclusion_text}")