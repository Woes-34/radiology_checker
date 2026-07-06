'''
定义数据类
用于表示影像报告中的发现项
用于表示影像报告的完整信息
'''
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

# 定义数据类
@dataclass
class Finding:
    anatomical_site: str # 体位(例如: 肺部, 胸部)
    lesion_type: str # 病灶类型(例如: 肺结节, 肺肿)
    polarity: bool # 极性，True为阳性（存在病变），False为阴性（不存在病变）
    qualifier: Optional[str] = None # 修饰符(例如: 可疑, 可能, 疑似, 考虑, 提示, 倾向, 轻微, 少量, 轻度)
    laterality: Optional[str] = None # 侧位(例如: 左, 右)
    raw_text: str = "" # 原始文本
    confidence: float = 1.0 # 置信度，0到1之间的浮点数，1表示完全信，0表示完全不信

@dataclass
# 存储完整报告信息
class Report:
    findings: str # 发现项的原始文本
    conclusion: str # 结论的原始文本
    findings_parsed: List[Finding] = field(default_factory=list) # 解析后的发现项列表
    conclusion_parsed: List[Finding] = field(default_factory=list) # 解析后的结论列表


@dataclass
# 存储矛盾结果
class ConflictResult:
    is_conflict: bool # 是否存在矛盾
    confidence: float # 矛盾的置信度
    explanation: str # 矛盾的解释
    source_pair: Tuple[Finding, Finding] # 矛盾的发现项对
    rule_type: str # 规则类型(例如: 逻辑规则, 语义规则)
