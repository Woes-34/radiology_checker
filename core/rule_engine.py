import re
import sys
import os
from typing import List, Tuple, Optional
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from radiology_checker.schema import Finding, ConflictResult
from radiology_checker.logger import get_logger


@dataclass
class MatchQuality:
    site_match: str  # 'exact', 'related', 'none'
    lesion_match: str  # 'exact', 'related', 'none'
    laterality_match: str  # 'same', 'different', 'unknown'
    polarity_match: str  # 'same', 'different'
    site_score: float
    lesion_score: float
    lat_score: float
    polarity_score: float
    total_score: float
    is_match: bool
    details: str


class RuleEngine:
    HIGH_CONFIDENCE_THRESHOLD = 0.8
    MEDIUM_CONFIDENCE_THRESHOLD = 0.55
    LOW_CONFIDENCE_THRESHOLD = 0.35
    # 初始化规则引擎，设置置信度阈值
    # 定义置信度等级对应的修饰符
    # 定义稳定、术后、治疗后等修饰符
    # 定义改进、好转等修饰符
    def __init__(self):
        self.logger = get_logger('RuleEngine')
        self.logger.info("初始化规则引擎")
        
        self.UNCERTAINTY_WORDS = {
            'high': ['可疑', '可能', '疑似'],
            'medium': ['考虑', '提示', '倾向于', '不除外'],
            'low': ['轻微', '少许', '少量', '轻度']
        }
        # 定义稳定、术后、治疗后等修饰符
        self.STABLE_WORDS = ['稳定', '不变', '无变化', '大小不变', '无进展', '未见进展']
        
        # 定义术后、治疗后等修饰符
        self.POST_OP_WORDS = ['术后', '切除后', '术后改变', '治疗后']
        
        # 定义改进、好转等修饰符    
        self.IMPROVEMENT_WORDS = ['吸收', '好转', '改善', '减轻', '消退']
        
    # 评估两个发现项匹配质量，根据解剖部位、病变类型、修饰符、侧位信息等计算匹配质量
    def _evaluate_match_quality(self, f1: Finding, f2: Finding) -> MatchQuality:
        quality = MatchQuality(
            site_match='none',
            lesion_match='none',
            laterality_match='unknown',
            polarity_match='same',
            site_score=0.0,
            lesion_score=0.0,
            lat_score=0.0,
            polarity_score=0.0,
            total_score=0.0,
            is_match=False,
            details=''
        )
        
        details = []
        
        # 评估解剖部位匹配质量
        if f1.anatomical_site == f2.anatomical_site:
            quality.site_match = 'exact'
            quality.site_score = 0.45
            details.append(f"部位精确匹配: {f1.anatomical_site}")
        elif self._is_site_related(f1.anatomical_site, f2.anatomical_site):
            quality.site_match = 'related'
            quality.site_score = 0.20
            details.append(f"部位相关匹配: {f1.anatomical_site} ↔ {f2.anatomical_site}")
        else:
            quality.site_match = 'none'
            quality.site_score = 0.0
            details.append(f"部位不匹配: {f1.anatomical_site} ≠ {f2.anatomical_site}")
            quality.details = ' | '.join(details)
            quality.is_match = False
            return quality
        
        is_negative = any(p in f1.raw_text for p in ['未见', '无', '没有']) or \
                     any(p in f2.raw_text for p in ['未见', '无', '没有'])
        
        # 评估病变类型匹配质量
        if f1.lesion_type == f2.lesion_type:
            quality.lesion_match = 'exact'
            quality.lesion_score = 0.35
            details.append(f"病变精确匹配: {f1.lesion_type}")
        elif self._is_lesion_related(f1.lesion_type, f2.lesion_type):
            quality.lesion_match = 'related'
            quality.lesion_score = 0.15
            details.append(f"病变相关匹配: {f1.lesion_type} ↔ {f2.lesion_type}")
        elif is_negative:
            negative_lesions = {'未见异常', '未见病变', '形态正常', '密度均匀', '正常', '未见明确异常', '未见明显异常', '无异常'}
            positive_groups = [
                {'结节', '肿块', '占位', '肿瘤'},
                {'炎症', '感染', '肺炎', '渗出'},
                {'纤维化', '钙化', '异常'},
                {'增大', '肿大', '扩张'},
                {'积液', '胸水', '心包积液'},
                {'栓塞', '血栓'},
                {'骨折', '骨裂', '骨破坏'},
                {'增厚', '肥厚'},
                {'水肿', '淤血'},
                {'空洞', '囊变'},
                {'退行性改变', '退变', '骨质增生'},
                {'病变', '病灶', '器质性'},
            ]
            pos_lesion = f1.lesion_type if f2.lesion_type in negative_lesions else f2.lesion_type
            neg_lesion = f2.lesion_type if f2.lesion_type in negative_lesions else f1.lesion_type
            for group in positive_groups:
                if pos_lesion in group:
                    quality.lesion_match = 'related'
                    quality.lesion_score = 0.15
                    details.append(f"病变极性相关: 阳性'{pos_lesion}' ↔ 阴性'{neg_lesion}'")
                    break
            else:
                if f1.lesion_type in ['异常', '病变', '器质性病变'] or f2.lesion_type in ['异常', '病变', '器质性病变']:
                    quality.lesion_match = 'related'
                    quality.lesion_score = 0.10
                    details.append(f"病变与'异常'相关: {f1.lesion_type} ↔ {f2.lesion_type}")
                else:
                    quality.lesion_match = 'none'
                    quality.lesion_score = 0.0
                    details.append(f"病变不匹配: {f1.lesion_type} ≠ {f2.lesion_type}")
                    quality.details = ' | '.join(details)
                    quality.is_match = False
                    return quality
        else:
            quality.lesion_match = 'none'
            quality.lesion_score = 0.0
            details.append(f"病变不匹配: {f1.lesion_type} ≠ {f2.lesion_type}")
            quality.details = ' | '.join(details)
            quality.is_match = False
            return quality
        
        # 评估侧位信息匹配质量
        if f1.laterality and f2.laterality:
            if f1.laterality == f2.laterality:
                quality.laterality_match = 'same'
                quality.lat_score = 0.08
                details.append(f"侧位一致: {f1.laterality}")
            else:
                quality.laterality_match = 'different'
                quality.lat_score = -0.05
                details.append(f"侧位不同: {f1.laterality} ≠ {f2.laterality}")
        elif f1.laterality or f2.laterality:
            quality.laterality_match = 'unknown'
            quality.lat_score = 0.0
            details.append(f"侧位信息缺失: {f1.laterality or '无'} / {f2.laterality or '无'}")
        else:
            quality.laterality_match = 'unknown'
            quality.lat_score = 0.0
            details.append("侧位信息缺失")
            
        # 评估极性匹配质量
        if f1.polarity == f2.polarity:
            quality.polarity_match = 'same'
            quality.polarity_score = 0.05
            details.append(f"极性一致: {'阳性' if f1.polarity else '阴性'}")
        else:
            quality.polarity_match = 'different'
            quality.polarity_score = -0.08
            details.append(f"极性不同: 所见{'阳性' if f1.polarity else '阴性'} vs 结论{'阳性' if f2.polarity else '阴性'}")

        # 评估不确定性惩罚
        uncertainty_penalty = self._calculate_uncertainty_penalty(f1.raw_text, f2.raw_text)
        if uncertainty_penalty > 0:
            details.append(f"不确定性惩罚: -{uncertainty_penalty:.2f}")
        # 评估稳定、术后、治疗后等修饰符
        if self._has_stable_pattern(f1.raw_text, f2.raw_text):
            quality.total_score -= 0.05
            details.append("检测到'稳定/不变'描述")
        # 评估术后、治疗后等修饰符
        if self._has_post_op_pattern(f1.raw_text, f2.raw_text):
            quality.total_score += 0.03
            details.append("检测到'术后改变'描述")
        # 评估好转、吸收等修饰符
        if self._has_improvement_pattern(f1.raw_text, f2.raw_text):
            details.append("检测到'好转/吸收'描述，降低矛盾优先级")
            if quality.polarity_match == 'different':
                quality.polarity_score += 0.08
        # 评估总分数
        quality.total_score = (
            quality.site_score + 
            quality.lesion_score + 
            quality.lat_score + 
            quality.polarity_score - 
            uncertainty_penalty
        )
        
        quality.is_match = quality.total_score >= 0.45
        
        quality.total_score = max(0.0, min(1.0, quality.total_score))
        
        quality.details = ' | '.join(details)
        
        return quality

    # 计算不确定性惩罚分数
    def _calculate_uncertainty_penalty(self, text1: str, text2: str) -> float:
        penalty = 0.0
        combined = text1 + text2
        
        for word in self.UNCERTAINTY_WORDS['high']:
            if word in combined:
                penalty += 0.15
                break
        
        for word in self.UNCERTAINTY_WORDS['medium']:
            if word in combined:
                penalty += 0.10
                break
        
        for word in self.UNCERTAINTY_WORDS['low']:
            if word in combined:
                penalty += 0.05
                break
        
        return min(penalty, 0.30)
    
    # 评估两个文本是否包含'稳定/不变'修饰符
    def _has_stable_pattern(self, text1: str, text2: str) -> bool:
        combined = text1 + text2
        return any(word in combined for word in self.STABLE_WORDS)
    
    # 评估两个文本是否包含'术后改变'修饰符
    def _has_post_op_pattern(self, text1: str, text2: str) -> bool:
        combined = text1 + text2
        return any(word in combined for word in self.POST_OP_WORDS)

    # 评估两个文本是否包含'好转/吸收'修饰符
    def _has_improvement_pattern(self, text1: str, text2: str) -> bool:
        combined = text1 + text2
        return any(word in combined for word in self.IMPROVEMENT_WORDS)

    # 判断两个剖析部位类型是否相关
    def _is_site_related(self, site1: str, site2: str) -> bool:
        related_groups = [
            {'肺', '胸腔', '胸膜', '肺门', '上叶', '下叶', '中叶'},
            {'心脏', '心包'},
            {'肝脏', '肝', '胆囊'},
            {'肾脏', '肾', '脾脏', '脾'},
            {'纵隔', '肺门'},
            {'支气管', '肺'},
            {'椎体', '腰椎', '胸椎', '颈椎', '脊柱'},
            {'骨骼', '骨', '骨质'},
            {'软组织', '肌肉', '肌腱', '韧带'},
            {'腹部', '腹腔', '腹膜后'},
            {'头颅', '颅内', '大脑', '小脑', '脑干', '基底节', '丘脑', '额叶', '颞叶', '枕叶'},
            {'盆腔', '膀胱', '前列腺', '子宫', '卵巢'},
            {'胆囊', '胆'},
        ]
        for group in related_groups:
            if site1 in group and site2 in group:
                return True
        return site1 == site2
    
    # 判断两个病变类型是否相关
    def _is_lesion_related(self, lesion1: str, lesion2: str) -> bool:
        related_groups = [
            {'结节', '肿块', '占位', '肿瘤'},
            {'炎症', '感染', '肺炎', '渗出'},
            {'纤维化', '钙化', '异常'},
            {'增大', '肿大', '扩张'},
            {'积液', '胸水', '心包积液'},
            {'栓塞', '血栓'},
            {'骨折', '骨裂', '骨破坏'},
            {'增厚', '肥厚'},
            {'水肿', '淤血'},
            {'空洞', '囊变'},
            {'术后改变', '变化', '改变'},
            {'退行性改变', '退变', '骨质增生'},
            {'异常', '病变', '病灶', '器质性'},
            {'形态正常', '密度均匀', '未见异常', '未见病变', '正常'},
            {'梗死', '脑梗死', '缺血', '低密度影'},
            {'出血', '血肿', '高密度影'},
            {'囊肿', '囊性占位'},
        ]
        for group in related_groups:
            if lesion1 in group and lesion2 in group:
                return True
        
        # 处理负向描述
        negative_lesions = {'未见异常', '未见病变', '形态正常', '密度均匀', '正常', '未见明确异常', '未见明显异常', '无异常'}
        if lesion1 in negative_lesions or lesion2 in negative_lesions:
            positive_groups = [
                {'结节', '肿块', '占位', '肿瘤'},
                {'炎症', '感染', '肺炎', '渗出'},
                {'纤维化', '钙化', '异常'},
                {'增大', '肿大', '扩张'},
                {'积液', '胸水', '心包积液'},
                {'栓塞', '血栓'},
                {'骨折', '骨裂', '骨破坏'},
                {'增厚', '肥厚'},
                {'水肿', '淤血'},
                {'空洞', '囊变'},
                {'退行性改变', '退变', '骨质增生'},
                {'病变', '病灶', '器质性'},
            ]
            non_neg = lesion1 if lesion2 in negative_lesions else lesion2
            for group in positive_groups:
                if non_neg in group:
                    return True
        
        return False
    
    # 判断两个侧位是是否冲突
    def _is_laterality_conflict(self, lat1: str, lat2: str) -> bool:
        lat1 = lat1.replace('侧', '').replace('肺', '').replace('胸', '')
        lat2 = lat2.replace('侧', '').replace('肺', '').replace('胸', '')
        
        if lat1 == '左' and lat2 == '右':
            return True
        if lat1 == '右' and lat2 == '左':
            return True
        
        if lat1 == '双' and lat2 in ['左', '右']:
            return False
        if lat2 == '双' and lat1 in ['左', '右']:
            return False
        
        exclusive_pairs = [('上', '下'), ('上', '中'), ('下', '中')]
        for a, b in exclusive_pairs:
            if (lat1 == a and lat2 == b) or (lat1 == b and lat2 == a):
                return True
        return False

    def _is_remainder_description(self, text: str) -> bool:
        remainder_words = ['其余', '其他', '余部', '余肺', '余肠', '余肝', '余肾', '余灶']
        return any(word in text for word in remainder_words)

    def _has_global_negation(self, text: str) -> bool:
        if self._is_remainder_description(text):
            return False
        
        global_negation_patterns = [
            r'未见异常', r'未见明显异常', r'未见明确异常',
            r'无异常', r'无明显异常', r'无明确异常',
            r'未见病变', r'未见明确病变',
            r'未见明显病变', r'未发现异常'
        ]
        
        for pattern in global_negation_patterns:
            if re.search(pattern, text):
                return True
        return False

    # 判断是否为限定性否定描述（包含'除.*外'或'未见.*'影'）
    def _is_qualified_negation(self, text: str) -> bool:
        if self._is_remainder_description(text):
            return True
        
        if re.search(r'除.*外', text):
            return True
        
        specific_negation_patterns = [
            r'未见明显异常密度影',
            r'未见明确异常密度影',
            r'未见异常密度影',
            r'未见明显异常强化',
            r'未见异常强化',
            r'未见明显异常信号',
            r'未见异常信号',
        ]
        
        for pattern in specific_negation_patterns:
            if pattern in text:
                return True
        
        return False

    # 匹配两个发现
    def _match_findings(self, f1: Finding, f2: Finding) -> Tuple[bool, float]:
        quality = self._evaluate_match_quality(f1, f2)
        return quality.is_match, quality.total_score
        
    # 检测极性冲突
    def _detect_polarity_conflict(self, f1: Finding, f2: Finding, 
                                   quality: MatchQuality) -> Optional[ConflictResult]:
        if quality.total_score < 0.45:
            return None
        
        if quality.lesion_match not in ['exact', 'related']:
            return None
        
        if quality.laterality_match == 'different':
            if f1.laterality == '双' or f2.laterality == '双':
                pass
            else:
                return None
        
        if quality.polarity_match != 'different':
            return None
        
        if self._should_skip_polarity_check(f1, f2):
            return None
        
        confidence = min(quality.total_score * 1.5, 0.95)
        
        if quality.lesion_match == 'exact':
            confidence = min(confidence + 0.10, 0.95)
        
        if quality.site_match == 'exact':
            confidence = min(confidence + 0.05, 0.95)
        
        uncertainty_penalty = self._calculate_uncertainty_penalty(f1.raw_text, f2.raw_text)
        confidence -= uncertainty_penalty
        
        if self._has_improvement_pattern(f1.raw_text, f2.raw_text):
            confidence -= 0.1
        
        confidence = max(0.5, min(0.95, confidence))
        
        explanation = f"极性冲突：所见'{f1.raw_text}'与结论'{f2.raw_text}'描述相反"
        return ConflictResult(
            is_conflict=True,
            confidence=confidence,
            explanation=explanation,
            source_pair=(f1, f2),
            rule_type='polarity_conflict'
        )
        
    # 判断是否应跳过极性检查
    def _should_skip_polarity_check(self, f1: Finding, f2: Finding) -> bool:
        combined = f1.raw_text + f2.raw_text
        
        if any(word in combined for word in self.STABLE_WORDS):
            return True
        
        if any(word in combined for word in self.POST_OP_WORDS):
            change_patterns = ['变化', '改变', '进展', '恶化']
            if not any(p in combined for p in change_patterns):
                return True
        
        if any(word in combined for word in self.IMPROVEMENT_WORDS):
            return True
        
        return False
        
    # 检测侧位冲突
    def _detect_laterality_conflict(self, f1: Finding, f2: Finding, 
                                    findings_list: List[Finding], 
                                    conclusion_list: List[Finding],
                                    quality: MatchQuality = None) -> Optional[ConflictResult]:
        if not f1.laterality or not f2.laterality:
            return None
        
        if not self._is_laterality_conflict(f1.laterality, f2.laterality):
            return None
        
        if f1.anatomical_site and f2.anatomical_site:
            if not self._is_site_related(f1.anatomical_site, f2.anatomical_site):
                return None
        
        if f1.lesion_type and f2.lesion_type:
            if not self._is_lesion_related(f1.lesion_type, f2.lesion_type):
                return None
        
        findings_laterality = set()
        for finding in findings_list:
            if finding.lesion_type == f1.lesion_type or self._is_lesion_related(finding.lesion_type, f1.lesion_type):
                if finding.laterality:
                    findings_laterality.add(finding.laterality)
        
        conclusion_laterality = set()
        for conclusion in conclusion_list:
            if conclusion.lesion_type == f2.lesion_type or self._is_lesion_related(conclusion.lesion_type, f2.lesion_type):
                if conclusion.laterality:
                    conclusion_laterality.add(conclusion.laterality)
        
        if (len(findings_laterality) >= 2 and '左' in findings_laterality and '右' in findings_laterality) or \
           (len(conclusion_laterality) >= 2 and '左' in conclusion_laterality and '右' in conclusion_laterality):
            return None
        
        if quality and quality.total_score >= 0.40:
            confidence = min(quality.total_score * 1.1, 0.95)
        else:
            confidence = 0.85
        
        confidence = max(0.3, min(0.95, confidence))
        
        explanation = f"侧位冲突：所见提及'{f1.laterality}'，结论提及'{f2.laterality}'"
        return ConflictResult(
            is_conflict=True,
            confidence=confidence,
            explanation=explanation,
            source_pair=(f1, f2),
            rule_type='laterality_conflict'
        )

    def _detect_site_subregion_conflict(self, f1: Finding, f2: Finding, 
                                        quality: MatchQuality) -> Optional[ConflictResult]:
        if not f1.raw_text or not f2.raw_text:
            return None
        
        if quality.total_score < 0.50:
            return None
        
        subregion_conflicts = [
            {'上叶', '下叶', '中叶'},
            {'前叶', '后叶', '侧叶'},
            {'上段', '下段', '中段'},
            {'近端', '远端', '中段'},
            {'内侧', '外侧', '前侧', '后侧'},
            {'上部', '下部', '中部'},
            {'左叶', '右叶', '尾叶', '方叶'},
            {'基底节', '丘脑', '额叶', '颞叶', '枕叶', '顶叶'},
            {'肾盂', '肾盏', '皮质', '髓质'},
            {'肝左叶', '肝右叶', '肝尾叶'},
        ]
        
        f1_subregions = []
        f2_subregions = []
        
        for region_set in subregion_conflicts:
            for region in region_set:
                if region in f1.raw_text:
                    f1_subregions.append((region, region_set))
                if region in f2.raw_text:
                    f2_subregions.append((region, region_set))
        
        for f1_region, f1_set in f1_subregions:
            for f2_region, f2_set in f2_subregions:
                if f1_set == f2_set and f1_region != f2_region:
                    if quality.lesion_match in ['exact', 'related']:
                        confidence = min(quality.total_score * 1.1, 0.95)
                        
                        explanation = f"部位子区域冲突：所见提及'{f1_region}'，结论提及'{f2_region}'"
                        return ConflictResult(
                            is_conflict=True,
                            confidence=confidence,
                            explanation=explanation,
                            source_pair=(f1, f2),
                            rule_type='site_subregion_conflict'
                        )
        
        return None
        
    # 检测限定词冲突
    def _detect_qualifier_conflict(self, f1: Finding, f2: Finding) -> Optional[ConflictResult]:
        if not f1.qualifier or not f2.qualifier:
            return None
        
        if self._is_qualifier_conflict(f1.qualifier, f2.qualifier):
            if f1.lesion_type != f2.lesion_type:
                return None
            
            explanation = f"限定词冲突：所见'{f1.qualifier}'与结论'{f2.qualifier}'矛盾"
            confidence = min(f1.confidence, f2.confidence) * 0.7
            return ConflictResult(
                is_conflict=True,
                confidence=confidence,
                explanation=explanation,
                source_pair=(f1, f2),
                rule_type='qualifier_conflict'
            )
        return None

    def _is_qualifier_conflict(self, q1: str, q2: str) -> bool:
        conflict_pairs = [
            ('明确', '可疑'),
            ('明显', '轻微'),
            ('大量', '少量'),
            ('多个', '单个'),
            ('广泛', '局部'),
            ('肯定', '可能'),
            ('严重', '轻微'),
        ]
        for a, b in conflict_pairs:
            if a in q1 and b in q2:
                return True
            if b in q1 and a in q2:
                return True
        return False
    
    # 检测阴性覆盖矛盾
    def _detect_negation_coverage_conflict(self, f: Finding, c: Finding,
                                           findings_list: List[Finding],
                                           conclusion_list: List[Finding]) -> Optional[ConflictResult]:
        if self._is_qualified_negation(f.raw_text) or self._is_qualified_negation(c.raw_text):
            return None
        
        negative_lesions = ['未见异常', '未见病变', '形态正常', '密度均匀', '异常']
        if f.lesion_type not in negative_lesions or f.polarity != False:
            return None
        
        if c.polarity != True:
            return None
        
        if f.anatomical_site == '全身' or self._is_site_related(f.anatomical_site, c.anatomical_site):
            if not f.laterality or f.laterality == c.laterality:
                confidence = min(f.confidence, c.confidence) * 0.85
                
                if '明确' in f.raw_text or '明显' in f.raw_text:
                    confidence *= 1.05
                
                confidence = min(0.95, confidence)
                
                explanation = f"阴性覆盖矛盾：所见'{f.raw_text}'与结论'{c.raw_text}'矛盾"
                return ConflictResult(
                    is_conflict=True,
                    confidence=confidence,
                    explanation=explanation,
                    source_pair=(f, c),
                    rule_type='negation_coverage_conflict'
                )
        
        return None
        
    # 检测显式否定矛盾
    def _detect_explicit_negation_conflict(self, f: Finding, c: Finding) -> Optional[ConflictResult]:
        if not f.raw_text or not c.raw_text:
            return None
        
        negation_patterns = [
            (['未见异常', '未见明显异常', '未见明确异常', '无异常'], ['结节', '肿块', '占位', '肿瘤', '病变', '骨折', '炎症', '积液', '增生', '突出', '膨出']),
            (['未见骨折', '未见骨质不连续'], ['骨折', '骨质不连续', '骨裂', '骨破坏']),
            (['未见结节'], ['结节', '肿块', '占位']),
            (['未见积液'], ['积液', '胸水', '心包积液']),
            (['未见增生'], ['增生', '骨质增生', '退行性改变']),
            (['未见突出'], ['突出', '膨出', '脱出']),
            (['未见炎症'], ['炎症', '感染', '肺炎']),
            (['未见异常密度影'], ['异常密度影', '低密度影', '高密度影']),
            (['未见扩张'], ['扩张']),
            (['未见狭窄'], ['狭窄']),
            (['未见增厚'], ['增厚']),
            (['未见移位'], ['移位']),
            (['未见肿大'], ['肿大', '增大']),
            (['未见钙化'], ['钙化']),
            (['未见损伤'], ['损伤', '撕裂']),
            (['未见脱位'], ['脱位']),
            (['未见梗死', '未见脑梗死'], ['梗死', '脑梗死', '低密度影', '缺血']),
            (['未见出血', '未见血肿'], ['出血', '血肿', '高密度影']),
            (['未见气胸'], ['气胸']),
        ]
        
        for negation_keywords, positive_keywords in negation_patterns:
            has_negation = any(neg in f.raw_text for neg in negation_keywords) or \
                          any(neg in c.raw_text for neg in negation_keywords)
            has_positive = any(pos in f.raw_text for pos in positive_keywords) or \
                          any(pos in c.raw_text for pos in positive_keywords)
            
            if has_negation and has_positive:
                f_has_neg = any(neg in f.raw_text for neg in negation_keywords)
                c_has_neg = any(neg in c.raw_text for neg in negation_keywords)
                f_has_pos = any(pos in f.raw_text for pos in positive_keywords)
                c_has_pos = any(pos in c.raw_text for pos in positive_keywords)
                
                if (f_has_neg and c_has_pos) or (f_has_pos and c_has_neg):
                    if f.anatomical_site and c.anatomical_site:
                        if not self._is_site_related(f.anatomical_site, c.anatomical_site):
                            return None
                    
                    confidence = 0.90
                    
                    if f.laterality and c.laterality:
                        if self._is_laterality_conflict(f.laterality, c.laterality):
                            return None
                    
                    if '可疑' in f.raw_text or '可疑' in c.raw_text:
                        confidence *= 0.85
                    elif '可能' in f.raw_text or '可能' in c.raw_text:
                        confidence *= 0.90
                    else:
                        confidence = min(confidence + 0.05, 0.95)
                    
                    neg_found = [neg for neg in negation_keywords if neg in f.raw_text or neg in c.raw_text]
                    pos_found = [pos for pos in positive_keywords if pos in f.raw_text or pos in c.raw_text]
                    
                    explanation = f"显式否定矛盾：{'所见' if f_has_neg else '结论'}提及'{neg_found[0]}'，{'结论' if c_has_pos else '所见'}提及'{pos_found[0]}'"
                    return ConflictResult(
                        is_conflict=True,
                        confidence=confidence,
                        explanation=explanation,
                        source_pair=(f, c),
                        rule_type='explicit_negation_conflict'
                    )
        
        return None
        
    # 检测推理矛盾（语义深层问题）
    def _detect_inferential_conflicts(self, findings_list: List[Finding], 
                                      conclusion_list: List[Finding],
                                      findings_raw_text: str = "",
                                      conclusion_raw_text: str = "") -> List[ConflictResult]:
        conflicts = []
        
        malignant_features = ['分叶', '毛刺', '胸膜凹陷', '血管集束', '不均匀强化', '边缘模糊', '浸润', '转移']
        benign_features = ['边界清晰', '边界清楚', '边缘光滑', '无毛刺', '形态规则', '密度均匀']
        malignant_conclusions = ['恶性肿瘤', '癌症', '癌', '恶性', '转移瘤', '浸润性癌', '肉瘤']
        benign_conclusions = ['良性', '囊肿', '血管瘤', '炎性假瘤', '肉芽肿', '良性囊肿', '良性肿瘤']
        
        findings_text = findings_raw_text if findings_raw_text else ' '.join([f.raw_text for f in findings_list])
        conclusion_text = conclusion_raw_text if conclusion_raw_text else ' '.join([c.raw_text for c in conclusion_list])
        
        no_abnormality_patterns = ['未见异常', '未见明显异常', '无异常']
        has_qualified_no_abnormality = False
        
        for pattern in no_abnormality_patterns:
            if pattern in findings_text:
                pattern_index = findings_text.find(pattern)
                before_text = findings_text[max(0, pattern_index - 10):pattern_index]
                if self._is_remainder_description(before_text + pattern):
                    has_qualified_no_abnormality = True
                    break
        
        has_benign_feature = any(feature in findings_text for feature in benign_features)
        
        benign_conclusion_text = any(conc in conclusion_text for conc in benign_conclusions)
        malignant_conclusion_text = any(conc in conclusion_text for conc in malignant_conclusions)
        
        has_malignant_conclusion = False
        for conc in malignant_conclusions:
            if conc in conclusion_text:
                has_malignant_conclusion = True
                break
        
        if not has_malignant_conclusion:
            if '可能性大' in conclusion_text or '可能大' in conclusion_text:
                if any(lesion in conclusion_text for lesion in ['结节', '肿块', '占位', '肿瘤']):
                    has_malignant_conclusion = True
        
        if has_benign_feature and has_malignant_conclusion:
            confidence = 0.75
            benign_count = sum(1 for feature in benign_features if feature in findings_text)
            if benign_count >= 2:
                confidence += 0.1
            
            explanation = f"推理矛盾：所见描述良性特征（{[f for f in benign_features if f in findings_text]}），但结论倾向恶性（{[c for c in malignant_conclusions if c in conclusion_text]}）"
            conflicts.append(ConflictResult(
                is_conflict=True,
                confidence=confidence,
                explanation=explanation,
                source_pair=(findings_list[0] if findings_list else None, conclusion_list[0] if conclusion_list else None),
                rule_type='inferential_conflict'
            ))
        
        has_malignant_feature = any(feature in findings_text for feature in malignant_features)
        has_benign_conclusion = any(conc in conclusion_text for conc in benign_conclusions)
        
        has_benign_direct = any(conc in findings_text for conc in benign_conclusions)
        has_malignant_direct = any(conc in conclusion_text for conc in malignant_conclusions)
        
        if has_benign_direct and has_malignant_direct:
            confidence = 0.85
            explanation = f"推理矛盾：所见描述良性病变（{[c for c in benign_conclusions if c in findings_text]}），但结论为恶性（{[c for c in malignant_conclusions if c in conclusion_text]}）"
            conflicts.append(ConflictResult(
                is_conflict=True,
                confidence=confidence,
                explanation=explanation,
                source_pair=(findings_list[0] if findings_list else None, conclusion_list[0] if conclusion_list else None),
                rule_type='inferential_conflict'
            ))
        
        has_malignant_direct_in_findings = any(conc in findings_text for conc in malignant_conclusions)
        has_benign_direct_in_conclusion = any(conc in conclusion_text for conc in benign_conclusions)
        
        if has_malignant_direct_in_findings and has_benign_direct_in_conclusion:
            confidence = 0.85
            explanation = f"推理矛盾：所见描述恶性病变（{[c for c in malignant_conclusions if c in findings_text]}），但结论为良性（{[c for c in benign_conclusions if c in conclusion_text]}）"
            conflicts.append(ConflictResult(
                is_conflict=True,
                confidence=confidence,
                explanation=explanation,
                source_pair=(findings_list[0] if findings_list else None, conclusion_list[0] if conclusion_list else None),
                rule_type='inferential_conflict'
            ))
        
        if has_malignant_feature and has_benign_conclusion:
            confidence = 0.75
            mal_count = sum(1 for feature in malignant_features if feature in findings_text)
            if mal_count >= 2:
                confidence += 0.1
            
            explanation = f"推理矛盾：所见描述恶性特征（{[f for f in malignant_features if f in findings_text]}），但结论倾向良性（{[c for c in benign_conclusions if c in conclusion_text]}）"
            conflicts.append(ConflictResult(
                is_conflict=True,
                confidence=confidence,
                explanation=explanation,
                source_pair=(findings_list[0] if findings_list else None, conclusion_list[0] if conclusion_list else None),
                rule_type='inferential_conflict'
            ))
        
        has_no_abnormality = any(neg in findings_text for neg in ['未见异常', '未见明显异常', '无异常'])
        
        all_qualified = True
        for pattern in ['未见异常', '未见明显异常', '无异常']:
            if pattern in findings_text:
                pattern_index = findings_text.find(pattern)
                before_text = findings_text[max(0, pattern_index - 10):pattern_index]
                if not self._is_remainder_description(before_text + pattern):
                    all_qualified = False
                    break
        
        if has_no_abnormality and not all_qualified:
            no_abnormality_sites = []
            for f in findings_list:
                if any(neg in f.raw_text for neg in ['未见异常', '未见明显异常', '无异常']):
                    if self._is_qualified_negation(f.raw_text):
                        continue
                    no_abnormality_sites.append(f.anatomical_site)
            
            for c in conclusion_list:
                if c.polarity == True:
                    site_match = False
                    for site in no_abnormality_sites:
                        if site == c.anatomical_site:
                            site_match = True
                            break
                        elif site == '全身':
                            site_match = True
                            break
                        elif self._is_site_related(site, c.anatomical_site):
                            site_match = True
                            break
                    
                    if site_match:
                        confidence = 0.80
                        uncertainty_words = ['可疑', '可能', '疑似', '考虑', '提示']
                        if any(w in findings_text or w in conclusion_text for w in uncertainty_words):
                            confidence *= 0.85
                        
                        explanation = f"推理矛盾：所见描述'未见异常'与结论'{c.raw_text}'在同一部位矛盾"
                        conflicts.append(ConflictResult(
                            is_conflict=True,
                            confidence=confidence,
                            explanation=explanation,
                            source_pair=(findings_list[0] if findings_list else None, c),
                            rule_type='inferential_conflict'
                        ))
        
        return conflicts

    # 主入口，检查所有冲突
    def check_conflicts(self, findings_list: List[Finding], 
                       conclusion_list: List[Finding],
                       findings_raw_text: str = "",
                       conclusion_raw_text: str = "") -> Tuple[List[ConflictResult], bool]:
        conflicts = []
        all_high_confidence = True
        
        self.logger.debug(f"开始冲突检测 - 所见: {len(findings_list)} 条, 结论: {len(conclusion_list)} 条")
        
        for f in findings_list:
            for c in conclusion_list:
                quality = self._evaluate_match_quality(f, c)
                
                if self._is_remainder_description(f.raw_text) or self._is_remainder_description(c.raw_text):
                    if self._is_remainder_description(f.raw_text) and self._is_remainder_description(c.raw_text):
                        pass
                    else:
                        continue
                
                neg_cover_conflict = self._detect_negation_coverage_conflict(f, c, findings_list, conclusion_list)
                if neg_cover_conflict:
                    conflicts.append(neg_cover_conflict)
                    if neg_cover_conflict.confidence < self.HIGH_CONFIDENCE_THRESHOLD:
                        all_high_confidence = False
                    continue
                
                explicit_neg_conflict = self._detect_explicit_negation_conflict(f, c)
                if explicit_neg_conflict:
                    conflicts.append(explicit_neg_conflict)
                    if explicit_neg_conflict.confidence < self.HIGH_CONFIDENCE_THRESHOLD:
                        all_high_confidence = False
                    continue
                
                if not quality.is_match:
                    continue
                
                polarity_conflict = self._detect_polarity_conflict(f, c, quality)
                if polarity_conflict:
                    conflicts.append(polarity_conflict)
                    if polarity_conflict.confidence < self.HIGH_CONFIDENCE_THRESHOLD:
                        all_high_confidence = False
                    continue
                
                lat_conflict = self._detect_laterality_conflict(f, c, findings_list, conclusion_list, quality)
                if lat_conflict:
                    conflicts.append(lat_conflict)
                    if lat_conflict.confidence < self.HIGH_CONFIDENCE_THRESHOLD:
                        all_high_confidence = False
                    continue
                
                subregion_conflict = self._detect_site_subregion_conflict(f, c, quality)
                if subregion_conflict:
                    conflicts.append(subregion_conflict)
                    if subregion_conflict.confidence < self.HIGH_CONFIDENCE_THRESHOLD:
                        all_high_confidence = False
                    continue
                
                qual_conflict = self._detect_qualifier_conflict(f, c)
                if qual_conflict:
                    conflicts.append(qual_conflict)
                    if qual_conflict.confidence < self.HIGH_CONFIDENCE_THRESHOLD:
                        all_high_confidence = False
        
        for f in findings_list:
            for c in conclusion_list:
                lat_conflict_direct = self._detect_laterality_conflict(f, c, findings_list, conclusion_list)
                if lat_conflict_direct:
                    exists = False
                    for existing in conflicts:
                        if existing.rule_type == 'laterality_conflict' and \
                           existing.source_pair[0] == f and existing.source_pair[1] == c:
                            exists = True
                            break
                    if not exists:
                        conflicts.append(lat_conflict_direct)
                        if lat_conflict_direct.confidence < self.HIGH_CONFIDENCE_THRESHOLD:
                            all_high_confidence = False
        
        inferential_conflicts = self._detect_inferential_conflicts(
            findings_list, conclusion_list,
            findings_raw_text, conclusion_raw_text
        )
        for inf_conflict in inferential_conflicts:
            conflicts.append(inf_conflict)
            if inf_conflict.confidence < self.HIGH_CONFIDENCE_THRESHOLD:
                all_high_confidence = False
        
        self.logger.debug(f"冲突检测完成 - 发现 {len(conflicts)} 个冲突, 全部高置信度: {all_high_confidence}")
        
        return conflicts, all_high_confidence

    def get_confidence_level(self, conflicts: List[ConflictResult]) -> str:
        if not conflicts:
            return 'high'
        
        max_confidence = max(c.confidence for c in conflicts)
        
        if max_confidence >= self.HIGH_CONFIDENCE_THRESHOLD:
            return 'high'
        elif max_confidence >= self.MEDIUM_CONFIDENCE_THRESHOLD:
            return 'medium'
        else:
            return 'low'
