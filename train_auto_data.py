import sys
import os
import json
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

random.seed(42)
np.random.seed(42)


def load_cleaned_reports(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def convert_to_nli_format(reports):
    nli_data = []
    for report in reports:
        findings = report.get('findings', '')
        conclusion = report.get('conclusion', '')
        if findings and conclusion:
            nli_data.append({
                'premise': findings,
                'hypothesis': conclusion,
                'label': 'entailment'
            })
    return nli_data


def generate_conflict_cases(reports, num_cases=50):
    conflict_cases = []
    
    negation_patterns = [
        (['未见异常', '未见明显异常', '未见明确异常'], '可见异常'),
        (['骨折', '骨质不连续'], '未见骨折'),
        (['结节', '肿块', '占位', '肿瘤'], '未见结节'),
        (['积液', '积气'], '未见积液'),
        (['增生', '退行性变'], '未见增生'),
        (['突出', '膨出', '脱出'], '未见突出'),
        (['炎症', '水肿'], '未见炎症'),
        (['低密度影', '异常密度影'], '未见异常密度影'),
        (['扩张'], '未见扩张'),
        (['狭窄'], '未见狭窄'),
        (['增厚'], '未见增厚'),
        (['移位'], '未见移位'),
        (['肿大'], '未见肿大'),
        (['钙化'], '未见钙化'),
        (['撕裂', '损伤'], '未见损伤'),
        (['脱位'], '未见脱位'),
        (['积液'], '未见积液'),
    ]
    
    for i in range(num_cases):
        report = random.choice(reports)
        findings = report.get('findings', '')
        conclusion = report.get('conclusion', '')
        
        if not findings or not conclusion:
            continue
        
        found_pattern = False
        for keywords, replacement in negation_patterns:
            for keyword in keywords:
                if keyword in conclusion:
                    new_conclusion = conclusion.replace(keyword, replacement, 1)
                    if new_conclusion != conclusion:
                        conflict_cases.append({
                            'premise': findings,
                            'hypothesis': new_conclusion,
                            'label': 'contradiction',
                            'original_conclusion': conclusion
                        })
                        found_pattern = True
                        break
            if found_pattern:
                break
        
        if not found_pattern:
            if '左侧' in conclusion:
                new_conclusion = conclusion.replace('左侧', '右侧', 1)
                conflict_cases.append({
                    'premise': findings,
                    'hypothesis': new_conclusion,
                    'label': 'contradiction',
                    'original_conclusion': conclusion
                })
            elif '右侧' in conclusion:
                new_conclusion = conclusion.replace('右侧', '左侧', 1)
                conflict_cases.append({
                    'premise': findings,
                    'hypothesis': new_conclusion,
                    'label': 'contradiction',
                    'original_conclusion': conclusion
                })
            elif '左' in conclusion and '右' in conclusion:
                new_conclusion = conclusion.replace('左', '右', 1).replace('右', '左', 1)
                conflict_cases.append({
                    'premise': findings,
                    'hypothesis': new_conclusion,
                    'label': 'contradiction',
                    'original_conclusion': conclusion
                })
            elif '术后改变' in conclusion:
                new_conclusion = conclusion.replace('术后改变', '未见异常')
                conflict_cases.append({
                    'premise': findings,
                    'hypothesis': new_conclusion,
                    'label': 'contradiction',
                    'original_conclusion': conclusion
                })
    
    return conflict_cases[:num_cases]


def split_dataset(data, train_ratio=0.7, val_ratio=0.2):
    random.shuffle(data)
    total = len(data)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    return train_data, val_data, test_data


def save_dataset(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def main():
    cleaned_reports = load_cleaned_reports('radiology_checker/data/cleaned_reports.json')
    print(f"原始报告数量: {len(cleaned_reports)}")
    
    nli_data = convert_to_nli_format(cleaned_reports)
    print(f"转换为NLI格式数量: {len(nli_data)}")
    
    conflict_cases = generate_conflict_cases(cleaned_reports, num_cases=5000)
    print(f"生成矛盾案例: {len(conflict_cases)}")
    
    all_data = nli_data + conflict_cases
    print(f"总数据量(正常+矛盾): {len(all_data)}")
    
    train_data, val_data, test_data = split_dataset(all_data)
    print(f"训练集: {len(train_data)}, 验证集: {len(val_data)}, 测试集: {len(test_data)}")
    
    save_dataset(train_data, 'radiology_checker/data/train_auto.jsonl')
    save_dataset(val_data, 'radiology_checker/data/val_auto.jsonl')
    save_dataset(test_data, 'radiology_checker/data/test_auto.jsonl')
    print("训练/验证/测试集已保存")
    
    train_entailment = sum(1 for d in train_data if d['label'] == 'entailment')
    train_contradiction = sum(1 for d in train_data if d['label'] == 'contradiction')
    print(f"\n训练集类别分布:")
    print(f"  entailment(不矛盾): {train_entailment}")
    print(f"  contradiction(矛盾): {train_contradiction}")


if __name__ == '__main__':
    main()
