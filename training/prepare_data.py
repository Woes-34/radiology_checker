import os
import sys
import json
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from radiology_checker.core.parser import FindingParser
from radiology_checker.core.rule_engine import RuleEngine


def generate_nli_data(reports, output_dir='radiology_checker/data', train_ratio=0.8, val_ratio=0.1):
    parser = FindingParser()
    rule_engine = RuleEngine()
    
    nli_data = []
    
    for report in reports:
        findings = report.get('findings', '')
        conclusion = report.get('conclusion', '')
        
        if not findings or not conclusion:
            continue
        
        findings_parsed = parser.parse(findings)
        conclusion_parsed = parser.parse(conclusion)
        
        if not findings_parsed or not conclusion_parsed:
            continue
        
        conflict_results, _ = rule_engine.check_conflicts(findings_parsed, conclusion_parsed)
        
        has_conflict = len(conflict_results) > 0
        
        if has_conflict:
            label = 'contradiction'
        else:
            similarity = rule_engine._evaluate_match_quality(findings_parsed[0], conclusion_parsed[0])
            if similarity.total_score >= 0.6:
                label = 'entailment'
            else:
                label = 'neutral'
        
        nli_data.append({
            'premise': findings,
            'hypothesis': conclusion,
            'label': label
        })
    
    random.seed(42)
    random.shuffle(nli_data)
    
    total = len(nli_data)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_data = nli_data[:train_end]
    val_data = nli_data[train_end:val_end]
    test_data = nli_data[val_end:]
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    def write_jsonl(data, filename):
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        return filepath
    
    train_file = write_jsonl(train_data, 'train.jsonl')
    val_file = write_jsonl(val_data, 'val.jsonl')
    test_file = write_jsonl(test_data, 'test.jsonl')
    
    print(f"生成NLI数据完成:")
    print(f"  总样本数: {total}")
    print(f"  训练集: {len(train_data)} ({len([d for d in train_data if d['label']=='contradiction'])}矛盾, {len([d for d in train_data if d['label']=='entailment'])}蕴含, {len([d for d in train_data if d['label']=='neutral'])}中立)")
    print(f"  验证集: {len(val_data)}")
    print(f"  测试集: {len(test_data)}")
    print(f"  训练文件: {train_file}")
    print(f"  验证文件: {val_file}")
    print(f"  测试文件: {test_file}")
    
    return train_file, val_file, test_file


if __name__ == '__main__':
    input_file = 'radiology_checker/data/cleaned_reports.json'
    if not os.path.exists(input_file):
        print(f"错误: 文件不存在 {input_file}")
        sys.exit(1)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        reports = json.load(f)
    
    print(f"加载报告数据: {len(reports)} 条")
    
    generate_nli_data(reports)