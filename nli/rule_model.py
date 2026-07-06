import sys
import os
import re
from typing import Optional, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from radiology_checker.nli.interface import NLIModelInterface, NLIResult
from radiology_checker.core.parser import FindingParser


class EnhancedRuleBasedNLI(NLIModelInterface):
    def __init__(self):
        self.parser = FindingParser()
        self.positive_keywords = {
            '骨折', '肿瘤', '结节', '肿块', '转移', '感染', '炎症', '积液', '出血',
            '梗死', '坏死', '扩张', '增厚', '增生', '钙化', '结石', '囊肿',
            '水肿', '脓肿', '溃疡', '穿孔', '狭窄', '畸形', '脱位', '撕裂',
            '疝', '血栓', '栓塞', '动脉瘤', '静脉曲张', '肝硬化', '脂肪肝',
            '肺炎', '肺结核', '肺气肿', '肺大泡', '胸腔积液', '心包积液',
            '胆囊炎', '胆结石', '胰腺炎', '肾结石', '肾盂积水', '膀胱结石',
            '前列腺增生', '子宫肌瘤', '卵巢囊肿', '宫颈癌', '乳腺癌',
            '甲状腺结节', '骨质增生', '椎间盘突出', '椎管狭窄', '退行性变',
            '脑膜瘤', '胶质瘤', '脑转移', '脑出血', '脑梗死', '脑积水',
            '硬膜下血肿', '蛛网膜下腔出血', '脑炎', '脑膜炎', '脊髓损伤',
            '败血症', '骨髓炎', '关节炎', '滑膜炎', '腱鞘炎', '软组织肿胀',
            '异常', '病变', '异常信号', '异常改变', '异常表现',
        }
        self.negative_keywords = {
            '未见', '未发现', '无明显', '无异常', '正常', '未见明显',
            '未见异常', '未显示', '未提示', '无特殊', '无异常发现',
            '未见明显异常', '无明显异常',
        }

    def _analyze_polarity(self, text: str) -> str:
        has_positive = any(kw in text for kw in self.positive_keywords)
        has_negative = any(kw in text for kw in self.negative_keywords)
        
        if has_negative and not has_positive:
            return 'negative'
        elif has_positive and not has_negative:
            return 'positive'
        elif has_positive and has_negative:
            if '未见' in text or '未发现' in text:
                return 'negative'
            return 'mixed'
        return 'neutral'

    def _extract_key_concepts(self, text: str) -> set:
        concepts = set()
        
        for kw in self.positive_keywords:
            if kw in text:
                concepts.add(kw)
        
        for site in self.parser.ANATOMICAL_SITES:
            if site in text:
                concepts.add(site)
        
        return concepts

    def predict(self, premise: str, hypothesis: str) -> NLIResult:
        premise_polarity = self._analyze_polarity(premise)
        hypothesis_polarity = self._analyze_polarity(hypothesis)
        
        premise_concepts = self._extract_key_concepts(premise)
        hypothesis_concepts = self._extract_key_concepts(hypothesis)
        
        common_concepts = premise_concepts & hypothesis_concepts
        
        if not common_concepts:
            return NLIResult(
                is_conflict=False,
                confidence=0.4,
                explanation="NLI判定：所见与结论无共同概念，无法判断"
            )
        
        if premise_polarity == 'positive' and hypothesis_polarity == 'negative':
            confidence = min(0.95, 0.5 + len(common_concepts) * 0.1)
            return NLIResult(
                is_conflict=True,
                confidence=confidence,
                explanation=f"NLI判定：所见描述阳性病变（{premise_polarity}），结论描述阴性（{hypothesis_polarity}），存在矛盾"
            )
        
        if premise_polarity == 'negative' and hypothesis_polarity == 'positive':
            confidence = min(0.95, 0.5 + len(common_concepts) * 0.1)
            return NLIResult(
                is_conflict=True,
                confidence=confidence,
                explanation=f"NLI判定：所见描述阴性（{premise_polarity}），结论描述阳性病变（{hypothesis_polarity}），存在矛盾"
            )
        
        if premise_polarity == hypothesis_polarity:
            confidence = min(0.9, 0.6 + len(common_concepts) * 0.05)
            return NLIResult(
                is_conflict=False,
                confidence=confidence,
                explanation=f"NLI判定：所见与结论极性一致（{premise_polarity}）"
            )
        
        if premise_polarity == 'mixed' or hypothesis_polarity == 'mixed':
            premise_findings = self.parser.parse(premise)
            hypothesis_findings = self.parser.parse(hypothesis)
            
            has_positive_in_premise = any(f.polarity for f in premise_findings)
            has_negative_in_hypothesis = any(not f.polarity for f in hypothesis_findings)
            
            if has_positive_in_premise and has_negative_in_hypothesis:
                return NLIResult(
                    is_conflict=True,
                    confidence=0.7,
                    explanation="NLI判定：所见包含阳性描述，结论包含阴性描述，存在矛盾"
                )
            
            return NLIResult(
                is_conflict=False,
                confidence=0.5,
                explanation="NLI判定：文本包含混合极性，无法确定"
            )
        
        return NLIResult(
            is_conflict=False,
            confidence=0.5,
            explanation="NLI判定：无法确定关系"
        )


class ContextualNLI(NLIModelInterface):
    def __init__(self):
        self.parser = FindingParser()
        
        self.site_hierarchy = {
            '腰椎': ['椎体', '脊柱'],
            '胸椎': ['椎体', '脊柱'],
            '颈椎': ['椎体', '脊柱'],
            '股骨': ['下肢', '骨骼'],
            '胫骨': ['下肢', '骨骼'],
            '肱骨': ['上肢', '骨骼'],
            '肝脏': ['腹部', '腹腔'],
            '肺部': ['胸腔', '肺'],
            '大脑': ['颅内', '头颅'],
            '小脑': ['颅内', '头颅'],
        }
        
        self.lesion_synonyms = {
            '退行性改变': ['退行性变', '退变', '骨质增生', '老化'],
            '器质性病变': ['病变', '异常', '病灶'],
            '骨折': ['骨裂', '骨断裂', '骨质中断'],
            '结节': ['肿块', '占位', '肿瘤'],
            '炎症': ['感染', '发炎'],
            '积液': ['积水', '渗出'],
            '出血': ['血肿'],
            '梗死': ['梗塞'],
            '扩张': ['增大', '增宽'],
            '增厚': ['肥厚'],
        }

    def _get_site_related(self, site: str) -> set:
        related = {site}
        for key, values in self.site_hierarchy.items():
            if site == key:
                related.update(values)
            elif site in values:
                related.add(key)
        return related

    def _get_lesion_related(self, lesion: str) -> set:
        related = {lesion}
        for key, values in self.lesion_synonyms.items():
            if lesion == key:
                related.update(values)
            elif lesion in values:
                related.add(key)
        return related

    def predict(self, premise: str, hypothesis: str) -> NLIResult:
        premise_findings = self.parser.parse(premise)
        hypothesis_findings = self.parser.parse(hypothesis)
        
        if not premise_findings and not hypothesis_findings:
            return NLIResult(
                is_conflict=False,
                confidence=0.3,
                explanation="NLI判定：无法解析文本"
            )
        
        conflicts_found = []
        for pf in premise_findings:
            for hf in hypothesis_findings:
                site_match = False
                if pf.anatomical_site and hf.anatomical_site:
                    pf_related = self._get_site_related(pf.anatomical_site)
                    hf_related = self._get_site_related(hf.anatomical_site)
                    site_match = len(pf_related & hf_related) > 0
                
                lesion_match = False
                if pf.lesion_type and hf.lesion_type:
                    pf_related = self._get_lesion_related(pf.lesion_type)
                    hf_related = self._get_lesion_related(hf.lesion_type)
                    lesion_match = len(pf_related & hf_related) > 0
                
                if site_match or lesion_match:
                    has_uncertainty = False
                    uncertainty_words = ['可疑', '可能', '疑似', '考虑', '提示', '倾向于']
                    if pf.qualifier and any(w in pf.qualifier for w in uncertainty_words):
                        has_uncertainty = True
                    if hf.qualifier and any(w in hf.qualifier for w in uncertainty_words):
                        has_uncertainty = True
                    
                    if pf.polarity != hf.polarity:
                        conflicts_found.append((pf, hf, has_uncertainty))
                    elif pf.polarity == hf.polarity and pf.laterality and hf.laterality:
                        if pf.laterality != hf.laterality:
                            conflicts_found.append((pf, hf, has_uncertainty))
        
        if conflicts_found:
            confidence = min(0.95, 0.6 + len(conflicts_found) * 0.1)
            pf, hf, has_uncertainty = conflicts_found[0]
            if has_uncertainty:
                confidence *= 0.7
            return NLIResult(
                is_conflict=True,
                confidence=confidence,
                explanation=f"NLI判定：所见'{pf.anatomical_site}{pf.lesion_type}'（{'阳性' if pf.polarity else '阴性'}）与结论'{hf.anatomical_site}{hf.lesion_type}'（{'阳性' if hf.polarity else '阴性'}）描述相反"
            )
        
        if premise_findings and hypothesis_findings:
            has_matching_site = False
            for pf in premise_findings:
                for hf in hypothesis_findings:
                    if pf.anatomical_site and hf.anatomical_site:
                        pf_related = self._get_site_related(pf.anatomical_site)
                        hf_related = self._get_site_related(hf.anatomical_site)
                        if len(pf_related & hf_related) > 0:
                            has_matching_site = True
                            break
                if has_matching_site:
                    break
            
            if has_matching_site:
                return NLIResult(
                    is_conflict=False,
                    confidence=0.75,
                    explanation="NLI判定：所见与结论描述一致"
                )
        
        return NLIResult(
            is_conflict=False,
            confidence=0.45,
            explanation="NLI判定：无匹配发现，无法确定关系"
        )
