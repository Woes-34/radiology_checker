# 完整评估指标计算
import sys
import os
import pandas as pd
sys.path.insert(0, r'D:\我的小工具')

from radiology_checker.core.main import ContradictionDetector
from radiology_checker.test_cases import TEST_CASES


def load_excel_test_cases(excel_path):
    """从Excel加载测试用例"""
    df = pd.read_excel(excel_path, sheet_name='Original report_中文翻译')
    df = df.iloc[1:].reset_index(drop=True)
    df.columns = ['报告编号', '检查类型', '检查信息', '影像所见', '诊断意见']
    
    cases = []
    for _, row in df.iterrows():
        cases.append({
            'name': f"报告{row['报告编号']}",
            'findings': str(row['影像所见']),
            'conclusion': str(row['诊断意见']),
            'expected': 'no_conflict'
        })
    return cases


def calculate_metrics(results):
    """计算评估指标"""
    # 二分类指标（忽略ambiguous）
    binary_results = [r for r in results if r['expected'] in ['conflict', 'no_conflict']]
    
    tp = sum(1 for r in binary_results if r['expected'] == 'conflict' and r['predicted'] == 'conflict')
    tn = sum(1 for r in binary_results if r['expected'] == 'no_conflict' and r['predicted'] == 'no_conflict')
    fp = sum(1 for r in binary_results if r['expected'] == 'no_conflict' and r['predicted'] == 'conflict')
    fn = sum(1 for r in binary_results if r['expected'] == 'conflict' and r['predicted'] == 'no_conflict')
    
    total_binary = len(binary_results)
    total_all = len(results)
    
    # 准确率
    accuracy = (tp + tn) / total_binary if total_binary > 0 else 0.0
    
    # 精确率（Precision）
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # 召回率（Recall）
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # F1分数
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # 宏平均准确率（包含ambiguous）
    correct_all = sum(1 for r in results if r['match'])
    overall_accuracy = correct_all / total_all if total_all > 0 else 0.0
    
    return {
        'total_all': total_all,
        'total_binary': total_binary,
        'correct_all': correct_all,
        'overall_accuracy': overall_accuracy,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def run_full_evaluation():
    detector = ContradictionDetector()
    
    # 1. 内置测试用例（20条）
    all_cases = []
    for case in TEST_CASES:
        all_cases.append(case)
    
    # 2. Excel测试用例（81条）
    excel_cases = load_excel_test_cases(r"C:\Users\24252\Desktop\医学影像报告翻译汇总（256份）.xlsx")
    all_cases.extend(excel_cases)
    
    print(f"总测试用例: {len(all_cases)} 条")
    print(f"  - 内置测试用例: {len(TEST_CASES)} 条")
    print(f"  - Excel测试用例: {len(excel_cases)} 条")
    
    # 运行测试
    results = []
    for case in all_cases:
        result = detector.analyze(case['findings'], case['conclusion'])
        pred = 'conflict' if result['is_conflict'] else 'no_conflict'
        results.append({
            'name': case['name'],
            'expected': case['expected'],
            'predicted': pred,
            'match': pred == case['expected']
        })
    
    # 计算指标
    metrics = calculate_metrics(results)
    
    # 输出结果
    print(f"\n{'='*60}")
    print(f"评估结果")
    print(f"{'='*60}")
    
    # 总体准确率
    print(f"\n📊 总体准确率（含边界案例）:")
    print(f"  总测试用例: {metrics['total_all']}")
    print(f"  正确: {metrics['correct_all']} ({metrics['overall_accuracy']*100:.2f}%)")
    print(f"  错误: {metrics['total_all'] - metrics['correct_all']} ({(1-metrics['overall_accuracy'])*100:.2f}%)")
    
    # 混淆矩阵
    print(f"\n📊 混淆矩阵（二分类，不含边界案例）:")
    print(f"{'':>12} 预测矛盾  预测不矛盾")
    print(f"{'实际矛盾':>12} {metrics['tp']:>10} {metrics['fn']:>12}")
    print(f"{'实际不矛盾':>12} {metrics['fp']:>10} {metrics['tn']:>12}")
    
    # 详细指标
    print(f"\n📊 详细评估指标（二分类）:")
    print(f"  {'准确率 (Accuracy):':<25} {metrics['accuracy']*100:.2f}%")
    print(f"  {'精确率 (Precision):':<25} {metrics['precision']*100:.2f}%")
    print(f"  {'召回率 (Recall):':<25} {metrics['recall']*100:.2f}%")
    print(f"  {'F1分数 (F1-score):':<25} {metrics['f1']*100:.2f}%")
    
    # 按类别统计
    print(f"\n📊 按预期类别统计:")
    expected_counts = {}
    for r in results:
        expected_counts[r['expected']] = expected_counts.get(r['expected'], 0) + 1
    
    for expected, count in expected_counts.items():
        correct = sum(1 for r in results if r['expected'] == expected and r['match'])
        rate = correct / count * 100
        print(f"  {expected}: {correct}/{count} ({rate:.2f}%)")
    
    # 显示错误案例
    errors = [r for r in results if not r['match']]
    if errors:
        print(f"\n❌ 错误案例 ({len(errors)}条):")
        for e in errors:
            print(f"  - {e['name']}: 预期={e['expected']}, 实际={e['predicted']}")


if __name__ == '__main__':
    run_full_evaluation()