import sys
import os
from typing import Optional, Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from radiology_checker.schema import Report, ConflictResult, Finding
from radiology_checker.core.parser import FindingParser
from radiology_checker.core.rule_engine import RuleEngine
from radiology_checker.logger import get_logger


class NLIResult:
    def __init__(self, is_conflict: bool, confidence: float, explanation: str, is_ambiguous: bool = False):
        self.is_conflict = is_conflict
        self.confidence = confidence
        self.explanation = explanation
        self.is_ambiguous = is_ambiguous

class NLIModelInterface:
    def predict(self, premise: str, hypothesis: str) -> NLIResult:
        raise NotImplementedError("Subclasses must implement predict method")


class RuleBasedNLIFallback(NLIModelInterface):
    def predict(self, premise: str, hypothesis: str) -> NLIResult:
        from radiology_checker.core.parser import FindingParser
        from radiology_checker.core.rule_engine import RuleEngine
        
        parser = FindingParser()
        rule_engine = RuleEngine()
        
        premise_findings = parser.parse(premise)
        hypothesis_findings = parser.parse(hypothesis)
        
        if not premise_findings and not hypothesis_findings:
            return NLIResult(
                is_conflict=False,
                confidence=0.35,
                explanation="NLI回退：无法解析文本"
            )
        
        conflicts, confidence_level = rule_engine.check_conflicts(
            premise_findings, hypothesis_findings
        )
        
        if conflicts:
            confidence = min(c.confidence for c in conflicts)
            return NLIResult(
                is_conflict=True,
                confidence=confidence,
                explanation=f"NLI回退：检测到{len(conflicts)}个矛盾"
            )
        else:
            quality_confidence = self._calculate_match_confidence(
                premise_findings, hypothesis_findings, rule_engine
            )
            return NLIResult(
                is_conflict=False,
                confidence=quality_confidence,
                explanation=f"NLI回退：未检测到矛盾 (匹配置信度: {quality_confidence:.2f})"
            )
    
    def _calculate_match_confidence(self, findings: List[Finding], 
                                    conclusions: List[Finding],
                                    rule_engine: RuleEngine) -> float:
        if not findings or not conclusions:
            return 0.40
        
        total_quality = 0.0
        matched_count = 0
        
        for f in findings:
            best_match = 0.0
            for c in conclusions:
                quality = rule_engine._evaluate_match_quality(f, c)
                if quality.is_match and quality.total_score > best_match:
                    best_match = quality.total_score
            
            if best_match > 0:
                total_quality += best_match
                matched_count += 1
        
        if matched_count == 0:
            has_related_site = False
            for f in findings:
                for c in conclusions:
                    if rule_engine._is_site_related(f.anatomical_site, c.anatomical_site):
                        has_related_site = True
                        break
                if has_related_site:
                    break
            return 0.50 if has_related_site else 0.40
        
        avg_quality = total_quality / matched_count
        coverage = matched_count / len(findings) if findings else 0
        
        confidence = avg_quality * (0.7 + 0.3 * coverage)
        
        if len(findings) > len(conclusions):
            ratio = len(conclusions) / len(findings)
            confidence *= (0.8 + 0.2 * ratio)
        
        return max(0.35, min(0.95, confidence))


class NLIManager:
    def __init__(self, model: Optional[NLIModelInterface] = None, use_neuro_model: bool = False):
        self.logger = get_logger('NLIManager')
        self.use_neuro_model = use_neuro_model
        self.neuro_model = None
        self.rule_engine = RuleEngine()
        
        if use_neuro_model:
            try:
                from radiology_checker.nli.neuro_model import NeuroNLIModel
                self.neuro_model = NeuroNLIModel()
                if self.neuro_model.is_available():
                    self.logger.info("神经NLI模型加载成功")
                else:
                    self.neuro_model = None
                    self.logger.warning("神经NLI模型不可用")
            except Exception as e:
                self.logger.error(f"神经NLI模型加载失败: {e}")
        
        self.model = model or RuleBasedNLIFallback()
        self.logger.info(f"NLI管理器初始化完成 - 使用神经NLI: {use_neuro_model}")
    
    def _calculate_no_conflict_confidence(self, report: Report) -> float:
        findings = report.findings_parsed
        conclusions = report.conclusion_parsed
        
        if not findings and not conclusions:
            return 0.35
        
        if not findings or not conclusions:
            return 0.45
        
        total_quality = 0.0
        matched_count = 0
        
        for f in findings:
            best_match = 0.0
            for c in conclusions:
                quality = self.rule_engine._evaluate_match_quality(f, c)
                if quality.is_match and quality.total_score > best_match:
                    best_match = quality.total_score
            
            if best_match > 0:
                total_quality += best_match
                matched_count += 1
        
        if matched_count == 0:
            has_related = False
            for f in findings:
                for c in conclusions:
                    if self.rule_engine._is_site_related(f.anatomical_site, c.anatomical_site):
                        has_related = True
                        break
                if has_related:
                    break
            return 0.50 if has_related else 0.40
        
        avg_quality = total_quality / matched_count
        coverage = matched_count / len(findings) if findings else 0
        
        confidence = avg_quality * (0.7 + 0.3 * coverage)
        
        if len(findings) > len(conclusions):
            ratio = len(conclusions) / len(findings)
            confidence *= (0.8 + 0.2 * ratio)
        
        return max(0.35, min(0.95, confidence))
    
    def analyze(self, report: Report, rule_conflicts: list, 
                confidence_level: str) -> Dict[str, Any]:
        self.logger.debug(f"NLI分析开始 - 置信度等级: {confidence_level}, 规则冲突数: {len(rule_conflicts)}")
        
        if confidence_level == 'high':
            if rule_conflicts:
                return {
                    'final_decision': 'rule_based',
                    'is_conflict': True,
                    'confidence': min(c.confidence for c in rule_conflicts),
                    'explanation': f"规则引擎检测到{len(rule_conflicts)}个矛盾",
                    'nli_used': False,
                    'is_ambiguous': False
                }
            else:
                no_conflict_confidence = self._calculate_no_conflict_confidence(report)
                
                if no_conflict_confidence < 0.60 and self.neuro_model and self.neuro_model.is_available():
                    self.logger.debug(f"匹配置信度低({no_conflict_confidence:.2f})，调用NLI模型作为第二意见")
                    findings_text = report.findings
                    conclusion_text = report.conclusion
                    nli_result = self.neuro_model.predict(findings_text, conclusion_text)
                    
                    final_is_conflict = nli_result.is_conflict
                    final_confidence = nli_result.confidence
                    is_ambiguous = False
                    
                    if nli_result.confidence >= 0.80 or nli_result.confidence <= 0.20:
                        is_ambiguous = False
                    elif 0.45 <= final_confidence <= 0.65:
                        is_ambiguous = True
                    
                    self.logger.debug(f"NLI第二意见完成 - 矛盾: {final_is_conflict}, 置信度: {final_confidence:.2f}")
                    
                    return {
                        'final_decision': 'neuro_nli',
                        'is_conflict': final_is_conflict,
                        'confidence': final_confidence,
                        'explanation': nli_result.explanation,
                        'nli_used': True,
                        'is_ambiguous': is_ambiguous
                    }
                else:
                    return {
                        'final_decision': 'rule_based',
                        'is_conflict': False,
                        'confidence': no_conflict_confidence,
                        'explanation': f"规则引擎未检测到矛盾 (匹配置信度: {no_conflict_confidence:.2f})",
                        'nli_used': False,
                        'is_ambiguous': no_conflict_confidence < 0.60
                    }
        
        findings_text = report.findings
        conclusion_text = report.conclusion
        
        nli_result = None
        model_used = 'rule_based'
        
        if self.neuro_model and self.neuro_model.is_available():
            nli_result = self.neuro_model.predict(findings_text, conclusion_text)
            model_used = 'neuro_nli'
        else:
            nli_result = self.model.predict(findings_text, conclusion_text)
        
        final_is_conflict = nli_result.is_conflict
        is_ambiguous = nli_result.is_ambiguous
        
        if rule_conflicts:
            rule_conflict = any(c.is_conflict for c in rule_conflicts)
            rule_max_confidence = max(c.confidence for c in rule_conflicts)
            
            if rule_conflict and nli_result.is_conflict:
                final_confidence = max(rule_max_confidence, nli_result.confidence) * 1.05
            elif rule_conflict != nli_result.is_conflict:
                if nli_result.confidence >= 0.80:
                    final_confidence = nli_result.confidence
                    final_is_conflict = nli_result.is_conflict
                    is_ambiguous = False
                else:
                    final_confidence = 0.50
                    final_is_conflict = nli_result.is_conflict
                    is_ambiguous = True
            else:
                final_confidence = nli_result.confidence
        else:
            final_confidence = nli_result.confidence
        
        final_confidence = min(1.0, max(0.0, final_confidence))
        
        if nli_result.confidence >= 0.80 or nli_result.confidence <= 0.20:
            is_ambiguous = False
        elif 0.45 <= final_confidence <= 0.65:
            is_ambiguous = True
        elif confidence_level == 'low' and final_confidence < 0.55:
            is_ambiguous = True
        
        return {
            'final_decision': model_used,
            'is_conflict': final_is_conflict,
            'confidence': final_confidence,
            'explanation': nli_result.explanation,
            'nli_used': True,
            'rule_conflicts': [c.explanation for c in rule_conflicts],
            'is_ambiguous': is_ambiguous
        }
