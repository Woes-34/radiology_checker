from typing import List, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from radiology_checker.schema import Report, Finding, ConflictResult
from radiology_checker.core.parser import FindingParser
from radiology_checker.core.rule_engine import RuleEngine
from radiology_checker.nli.interface import NLIManager, NLIModelInterface
from radiology_checker.config import get_config
from radiology_checker.logger import get_logger


class ContradictionDetector:
    def __init__(self, nli_model: NLIModelInterface = None, use_neuro_model: bool = False):
        self.config = get_config()
        self.logger = get_logger('ContradictionDetector')
        self.logger.info("初始化矛盾检测系统")
        
        self.parser = FindingParser()
        self.rule_engine = RuleEngine()
        
        use_neuro = use_neuro_model or self.config.get('nli.enable_neuro_nli', True)
        self.nli_manager = NLIManager(nli_model, use_neuro_model=use_neuro)
        self.logger.info(f"NLI模型配置: use_neuro_model={use_neuro}")
        
    # 处理报告，解析文本为Finding对象
    def process_report(self, findings_text: str, conclusion_text: str) -> Report:
        report = Report(findings=findings_text, conclusion=conclusion_text)
        
        report.findings_parsed = self.parser.parse(findings_text)
        report.conclusion_parsed = self.parser.parse(conclusion_text)
        
        return report
        
    # 检测矛盾
    def detect_conflicts(self, report: Report) -> Tuple[List[ConflictResult], str]:
        conflicts, all_high_confidence = self.rule_engine.check_conflicts(
            report.findings_parsed,
            report.conclusion_parsed,
            report.findings,
            report.conclusion
        )
        
        confidence_level = self.rule_engine.get_confidence_level(conflicts)
        
        return conflicts, confidence_level
        
    # 分析报告，包括规则引擎检测和NLI分析
    def analyze(self, findings_text: str, conclusion_text: str) -> dict:
        self.logger.debug(f"开始分析报告 - 所见: {findings_text[:50]}... 结论: {conclusion_text[:50]}...")
        
        report = self.process_report(findings_text, conclusion_text)
        self.logger.debug(f"解析完成 - 所见: {len(report.findings_parsed)} 条, 结论: {len(report.conclusion_parsed)} 条")
        
        conflicts, confidence_level = self.detect_conflicts(report)
        self.logger.debug(f"规则引擎检测完成 - 冲突数: {len(conflicts)}, 置信度等级: {confidence_level}")
        
        nli_result = self.nli_manager.analyze(report, conflicts, confidence_level)
        
        result = {
            'findings_parsed': report.findings_parsed,
            'conclusion_parsed': report.conclusion_parsed,
            'conflicts': conflicts,
            'confidence_level': confidence_level,
            'needs_nli': confidence_level != 'high',
            'is_conflict': nli_result['is_conflict'],
            'is_ambiguous': nli_result.get('is_ambiguous', False),
            'final_confidence': nli_result['confidence'],
            'decision_source': nli_result['final_decision'],
            'nli_used': nli_result['nli_used'],
            'nli_explanation': nli_result['explanation']
        }
        
        conflict_str = '存在逻辑矛盾' if result['is_conflict'] else '未发现逻辑矛盾'
        if result.get('is_ambiguous'):
            conflict_str = '边界案例（需要人工审核）'
        
        self.logger.info(f"分析完成 - 结论: {conflict_str}, 置信度: {result['final_confidence']:.2f}, "
                        f"判定来源: {result['decision_source']}, NLI使用: {result['nli_used']}")
        
        return result

# 格式化输出结果
def format_result(result: dict) -> str:
    output = []
    
    output.append("=" * 60)
    output.append("影像报告逻辑矛盾检测结果")
    output.append("=" * 60)
    
    output.append("\n【所见】解析结果:")
    for i, finding in enumerate(result['findings_parsed'], 1):
        polarity_str = '阳性' if finding.polarity else '阴性'
        lat_str = f"侧位:{finding.laterality}" if finding.laterality else ""
        qual_str = f"限定词:{finding.qualifier}" if finding.qualifier else ""
        output.append(f"  {i}. 部位:{finding.anatomical_site} 病变:{finding.lesion_type} "
                      f"极性:{polarity_str} {lat_str} {qual_str} "
                      f"置信度:{finding.confidence:.2f}")
    
    output.append("\n【结论】解析结果:")
    for i, finding in enumerate(result['conclusion_parsed'], 1):
        polarity_str = '阳性' if finding.polarity else '阴性'
        lat_str = f"侧位:{finding.laterality}" if finding.laterality else ""
        qual_str = f"限定词:{finding.qualifier}" if finding.qualifier else ""
        output.append(f"  {i}. 部位:{finding.anatomical_site} 病变:{finding.lesion_type} "
                      f"极性:{polarity_str} {lat_str} {qual_str} "
                      f"置信度:{finding.confidence:.2f}")
    
    output.append("\n【规则引擎冲突检测】:")
    if result['conflicts']:
        for i, conflict in enumerate(result['conflicts'], 1):
            output.append(f"  {i}. [矛盾] {conflict.explanation}")
            output.append(f"     置信度: {conflict.confidence:.2f}")
            output.append(f"     规则类型: {conflict.rule_type}")
    else:
        output.append("  未检测到逻辑矛盾")
    
    output.append(f"\n【规则置信度等级】: {result['confidence_level']}")
    output.append(f"【是否使用NLI模型】: {'是' if result['nli_used'] else '否'}")
    
    source_map = {
        'rule_based': '规则引擎',
        'nli_based': '规则NLI模型',
        'neuro_nli': '神经NLI模型'
    }
    output.append(f"【最终判定来源】: {source_map.get(result['decision_source'], result['decision_source'])}")
    
    if result.get('is_ambiguous', False):
        output.append(f"【最终结论】: 边界案例（需要人工审核）")
    else:
        output.append(f"【最终结论】: {'存在逻辑矛盾' if result['is_conflict'] else '未发现逻辑矛盾'}")
    
    output.append(f"【最终置信度】: {result['final_confidence']:.2f}")
    
    if result['nli_used']:
        output.append(f"【NLI模型解释】: {result['nli_explanation']}")
    
    output.append("\n" + "=" * 60)
    
    return "\n".join(output)

# 运行测试用例
def run_test_cases():
    detector = ContradictionDetector()
    
    print("运行测试用例...\n")
    
    for case in TEST_CASES:
        print(f"测试用例: {case['name']}")
        print("-" * 40)
        
        result = detector.analyze(case['findings'], case['conclusion'])
        
        print(f"所见: {case['findings']}")
        print(f"结论: {case['conclusion']}")
        print(f"预期: {case['expected']}")
        print(f"检测: {'矛盾' if result['is_conflict'] else '不矛盾'}")
        print(f"置信度: {result['confidence_level']}")
        
        if result['conflicts']:
            for c in result['conflicts']:
                print(f"  - {c.explanation}")
        
        print()

        
# 交互式模式
def interactive_mode():
    detector = ContradictionDetector()
    
    print("=" * 60)
    print("影像报告逻辑矛盾检测系统 v1.0")
    print("=" * 60)
    print()
    
    while True:
        print("请输入影像报告内容：")
        findings = input("【所见】: ").strip()
        if not findings:
            print("所见内容不能为空，请重新输入")
            continue
        
        conclusion = input("【结论】: ").strip()
        if not conclusion:
            print("结论内容不能为空，请重新输入")
            continue
        
        result = detector.analyze(findings, conclusion)
        print()
        print(format_result(result))
        print()
        
        choice = input("继续检测？(y/n): ").strip().lower()
        if choice != 'y':
            break
    
    print("感谢使用！")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        from radiology_checker.tests.test_cases import TEST_CASES
        run_test_cases()
    else:
        interactive_mode()
