import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.rule_engine import RuleEngine
from core.parser import FindingParser
from nli.neuro_model import NeuroNLIModel


def load_test_data(file_path, limit=50):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def main():
    test_data = load_test_data('radiology_checker/data/test.jsonl', limit=50)
    
    rule_engine = RuleEngine()
    parser = FindingParser()
    nli_model = NeuroNLIModel()
    
    print(f"{'='*90}")
    print(f"{'测试集联合检测结果':^90}")
    print(f"{'='*90}")
    print(f"{'编号':<5} {'规则引擎':<12} {'规则置信度':<10} {'NLI模型':<12} {'NLI置信度':<10} {'真实标签':<15} {'一致':<6}")
    print(f"{'='*90}")
    
    results = []
    for i, item in enumerate(test_data, 1):
        findings_text = item['premise']
        conclusion_text = item['conclusion'] if 'conclusion' in item else item['hypothesis']
        true_label = item['label']
        
        findings_parsed = parser.parse(findings_text)
        conclusion_parsed = parser.parse(conclusion_text)
        
        rule_conflicts, _ = rule_engine.check_conflicts(findings_parsed, conclusion_parsed, findings_text, conclusion_text)
        rule_result = '矛盾' if rule_conflicts else '不矛盾'
        rule_confidence = max(c.confidence for c in rule_conflicts) if rule_conflicts else 0.0
        
        nli_result_obj = nli_model.predict(findings_text, conclusion_text)
        nli_result = '矛盾' if nli_result_obj.is_conflict else '不矛盾'
        nli_confidence = nli_result_obj.confidence
        
        combined_result = '矛盾' if rule_result == '矛盾' or nli_result == '矛盾' else '不矛盾'
        
        true_result = '矛盾' if true_label == 'contradiction' else '不矛盾'
        is_match = '✓' if combined_result == true_result else '✗'
        
        print(f"{i:<5} {rule_result:<12} {rule_confidence:<10.4f} {nli_result:<12} {nli_confidence:<10.4f} {true_label:<15} {is_match:<6}")
        
        results.append({
            'id': i,
            'rule_result': rule_result,
            'rule_confidence': rule_confidence,
            'nli_result': nli_result,
            'nli_confidence': nli_confidence,
            'true_label': true_label,
            'combined_result': combined_result,
            'is_match': is_match == '✓'
        })
    
    print(f"{'='*90}")
    
    correct = sum(1 for r in results if r['is_match'])
    accuracy = correct / len(results) * 100
    
    rule_correct = sum(1 for r in results if (r['rule_result'] == '矛盾') == (r['true_label'] == 'contradiction'))
    rule_accuracy = rule_correct / len(results) * 100
    
    nli_correct = sum(1 for r in results if (r['nli_result'] == '矛盾') == (r['true_label'] == 'contradiction'))
    nli_accuracy = nli_correct / len(results) * 100
    
    print(f"\n📊 统计结果:")
    print(f"{'='*60}")
    print(f"总测试样本: {len(results)}")
    print(f"规则引擎准确率: {rule_accuracy:.2f}% ({rule_correct}/{len(results)})")
    print(f"NLI模型准确率: {nli_accuracy:.2f}% ({nli_correct}/{len(results)})")
    print(f"联合检测准确率: {accuracy:.2f}% ({correct}/{len(results)})")
    
    tp = sum(1 for r in results if r['true_label'] == 'contradiction' and r['combined_result'] == '矛盾')
    tn = sum(1 for r in results if r['true_label'] != 'contradiction' and r['combined_result'] == '不矛盾')
    fp = sum(1 for r in results if r['true_label'] != 'contradiction' and r['combined_result'] == '矛盾')
    fn = sum(1 for r in results if r['true_label'] == 'contradiction' and r['combined_result'] == '不矛盾')
    
    print(f"\n📊 混淆矩阵:")
    print(f"{'='*60}")
    print(f"{'':>12} 预测矛盾  预测不矛盾")
    print(f"{'实际矛盾':>12} {tp:>10} {fn:>12}")
    print(f"{'实际不矛盾':>12} {fp:>10} {tn:>12}")
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n📊 详细指标:")
    print(f"{'='*60}")
    print(f"精确率 (Precision): {precision*100:.2f}%")
    print(f"召回率 (Recall): {recall*100:.2f}%")
    print(f"F1分数 (F1-score): {f1*100:.2f}%")


if __name__ == '__main__':
    main()
