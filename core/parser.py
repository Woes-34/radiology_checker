import re
import sys
import os
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from radiology_checker.schema import Finding
from radiology_checker.logger import get_logger

class FindingParser:
    NEGATION_PATTERNS = [
        r'未见', r'未发现', r'无', r'没有', r'缺乏',
        r'不存在', r'未显示', r'未检出', r'未提示',
        r'不能排除', r'不除外', r'不考虑', r'可能不大',
        r'难以确定', r'尚不明确', r'不典型',
        r'已吸收', r'已消退', r'已好转', r'已消失',
        r'完全吸收', r'明显吸收', r'明显好转', r'较前吸收',
        r'较前好转', r'较前改善',
    ]

    LATERALITY_PATTERNS = {
        '左': [
            r'左肺', r'左侧', r'左胸', r'左上', r'左下', r'左叶',
            r'左肾', r'左输尿管', r'左肾上腺',
            r'左甲状腺', r'左乳腺', r'左卵巢', r'左输卵管',
            r'左肩', r'左髋', r'左膝', r'左肘', r'左腕', r'左踝',
            r'左股骨', r'左胫骨', r'左腓骨', r'左肱骨', r'左尺骨', r'左桡骨',
            r'左肋骨', r'左锁骨', r'左肩胛骨',
            r'左肝', r'左脾',
            r'左冠状动脉', r'左心房', r'左心室',
            r'左半'
        ],
        '右': [
            r'右肺', r'右侧', r'右胸', r'右上', r'右下', r'右叶',
            r'右肾', r'右输尿管', r'右肾上腺',
            r'右甲状腺', r'右乳腺', r'右卵巢', r'右输卵管',
            r'右肩', r'右髋', r'右膝', r'右肘', r'右腕', r'右踝',
            r'右股骨', r'右胫骨', r'右腓骨', r'右肱骨', r'右尺骨', r'右桡骨',
            r'右肋骨', r'右锁骨', r'右肩胛骨',
            r'右肝', r'右脾',
            r'右冠状动脉', r'右心房', r'右心室',
            r'右半', r'右后叶', r'右前叶', r'右内叶', r'右外叶',
        ],
        '双': [r'双肺', r'双侧', r'两肺', r'两侧', r'双肾', r'双乳腺'],
        '中': [r'中叶', r'中间'],
        '上': [r'上叶', r'上部', r'上段'],
        '下': [r'下叶', r'下部', r'下段']
    }

    ANATOMICAL_SITES = [
        '肺', '胸腔', '纵隔', '胸膜', '心包', '心脏', '气管', '支气管',
        '肝脏', '脾脏', '肾脏', '胆囊', '胰腺', '胃', '肠道', '阑尾', '腹部', '胰', '十二指肠', '肝', '脾',
        '主动脉', '回肠', '结肠', '直肠', '空肠', '小肠', '大肠', '食管', '胃底', '贲门',
        '腹膜后', '盆腔', '纵隔', '胸膜', '心包', '腹腔', '盆腔', '脊柱', '椎管',
        '头颅', '颅骨', '颞骨', '头皮', '颅内', '大脑', '小脑', '脑干', '脑室', '脑膜',
        '基底节', '基底节区', '鞍区', '垂体', '松果体', '丘脑', '海马', '胼胝体',
        '中脑', '导水管', '颞叶', '额叶', '顶叶', '枕叶', '小脑半球', '脑室系统',
        '脑白质', '脑灰质', '海马体', '杏仁核', '豆状核', '尾状核', '内囊', '外囊',
        '硬膜下', '硬膜外', '蛛网膜下腔', '脑室周围', '额颞顶', '枕叶', '小脑蚓部',
        '骨骼', '关节', '肋骨', '椎骨', '椎体', '骨盆', '肩胛骨', '锁骨',
        '股骨', '胫骨', '腓骨', '肱骨', '尺骨', '桡骨', '髌骨',
        '脊柱', '颈椎', '胸椎', '腰椎', '骶椎', '尾椎',
        '血管', '淋巴结', '软组织', '肌肉', '肌腱', '韧带', '筋膜',
        '中央', '中央型',
        '小叶', '小叶间隔',
        '肺门', '肺野', '肺实质',
        '椎间盘', '椎管',
        '肾', '脾',
        '甲状腺', '肾上腺',
        '乳腺', '卵巢', '输卵管', '子宫', '前列腺',
        '输尿管', '膀胱',
        '肩关节', '髋关节', '膝关节', '肘关节', '腕关节', '踝关节',
    ]

    BONE_SPECIFIC_PATTERNS = [
        r'股骨颈', r'股骨头', r'股骨干',
        r'胫骨平台', r'胫骨干', r'胫骨结节',
        r'肱骨头', r'肱骨干', r'肱骨髁',
        r'髋臼', r'髂骨', r'坐骨', r'耻骨',
        r'肩关节', r'肘关节', r'腕关节', r'膝关节', r'踝关节',
        r'椎体', r'椎间盘', r'椎管', r'椎弓根',
        r'骨盆环', r'骶髂关节'
    ]

    LESION_TYPES = [
        ('结节', ['结节', '小结节', '微结节', '结节影']),
        ('肿块', ['肿块', '占位', '占位病变']),
        ('积液', ['积液', '胸水', '胸腔积液', '心包积液']),
        ('炎症', ['炎症', '炎性', '肺炎', '感染']),
        ('纤维化', ['纤维化', '纤维灶', '条索影']),
        ('钙化', ['钙化', '钙化灶']),
        ('空洞', ['空洞', '空洞影']),
        ('扩张', ['扩张', '支气管扩张']),
        ('增厚', ['增厚', '胸膜增厚']),
        ('增大', ['增大', '肿大', '体积增大']),
        ('骨折', ['骨折', '骨裂', '粉碎性骨折', '压缩性骨折']),
        ('移位', ['移位', '脱位']),
        ('狭窄', ['狭窄', '变窄']),
        ('出血', ['出血', '血肿']),
        ('水肿', ['水肿']),
        ('渗出', ['渗出', '渗出性']),
        ('阴影', ['阴影', '片状阴影', '斑片状阴影']),
        ('磨玻璃影', ['磨玻璃影', '磨玻璃密度', '磨玻璃样改变', 'GGO']),
        ('肺栓塞', ['肺栓塞', '肺动脉栓塞', 'PE', '栓塞']),
        ('退行性改变', ['退行性改变', '退变', '骨质增生', '骨赘']),
        ('骨质破坏', ['骨质破坏', '骨破坏', '溶骨性破坏']),
        ('斑片影', ['斑片影', '斑片状影', '片状影']),
        ('小叶间隔增厚', ['小叶间隔增厚', '间隔增厚']),
        ('肺淤血', ['肺淤血', '肺静脉淤血', '淤血']),
        ('浸润性病变', ['浸润性病变', '浸润影', '浸润']),
        ('伪影', ['伪影', '运动伪影', '造影伪影']),
        ('造影淡染', ['造影淡染', '造影欠佳']),
        ('术后改变', ['术后改变', '术后', '术后瘢痕', '术后残留', '切除术后']),
        ('陈旧性病变', ['陈旧性', '陈旧性病变', '陈旧性改变', '陈旧性病灶']),
        ('肺不张', ['肺不张', '不张', '肺膨胀不全']),
        ('肺气肿', ['肺气肿', '气肿', '肺大泡']),
        ('肺实变', ['肺实变', '实变', '完全不张']),
        ('心脏扩大', ['心脏扩大', '心腔扩大', '心腔明显扩大']),
        ('纵隔移位', ['纵隔移位', '纵隔偏移', '纵隔明显移位']),
        ('肺栓塞', ['肺栓塞', '肺动脉栓塞', '栓塞']),
        ('心包积液', ['心包积液']),
        ('胸膜增厚', ['胸膜增厚']),
        ('变化', ['变化', '改变', '明显变化', '异常变化']),
        ('团块', ['团块', '团块影']),
        ('分叶状', ['分叶状', '分叶']),
        ('边界清晰', ['边界清晰', '边界清楚', '边缘清晰', '边缘清楚']),
        ('边缘模糊', ['边缘模糊', '边界模糊']),
        ('毛刺', ['毛刺', '毛刺征']),
        ('胸膜凹陷', ['胸膜凹陷', '胸膜凹陷征']),
        ('血管集束', ['血管集束', '血管集束征']),
        ('纹理增多', ['纹理增多', '肺纹理增多']),
        ('模糊', ['模糊']),
        ('支气管炎', ['支气管炎']),
        ('周围型肺癌', ['周围型肺癌', '肺癌', '肺恶性肿瘤']),
        ('中央型肺癌', ['中央型肺癌']),
        ('肺癌', ['肺癌']),
        ('可能性大', ['可能性大', '可能大']),
        ('软组织肿胀', ['软组织肿胀', '肿胀']),
        ('结石', ['结石', '钙化灶', '高密度影']),
        ('囊性占位', ['囊性占位', '囊肿', '囊性病变']),
        ('低回声结节', ['低回声结节', '低回声']),
        ('骨质增生', ['骨质增生', '增生']),
        ('转移灶', ['转移灶', '转移', '转移瘤', '转移癌']),
        ('器质性病变', ['器质性病变', '器质性']),
        ('异常', ['异常', '异常信号', '异常改变', '异常表现']),
        ('病灶', ['病灶', '病变', '占位病变', '占位']),
        ('肿块影', ['肿块影', '软组织肿块']),
        ('结节影', ['结节影', '软组织结节']),
        ('高密度影', ['高密度影', '高密度']),
        ('低密度影', ['低密度影', '低密度']),
        ('异常强化', ['异常强化', '强化']),
        ('缺血', ['缺血', '脑缺血', '心肌缺血']),
        ('梗死', ['梗死', '脑梗死', '心肌梗死', '肺梗死']),
        ('萎缩', ['萎缩', '脑萎缩', '肌肉萎缩']),
        ('肥大', ['肥大', '心肌肥大']),
        ('狭窄', ['狭窄', '管腔狭窄', '血管狭窄']),
        ('扩张', ['扩张', '血管扩张']),
        ('畸形', ['畸形', '先天性畸形']),
        ('缺损', ['缺损', '骨质缺损']),
        ('坏死', ['坏死', '骨坏死', '缺血性坏死']),
        ('炎症', ['炎症', '炎性病变']),
        ('肝硬化', ['肝硬化', '肝硬变']),
        ('门静脉高压', ['门静脉高压', '门脉高压']),
        ('静脉曲张', ['静脉曲张', '曲张']),
        ('腹水', ['腹水', '腹腔积液']),
        ('肝细胞癌', ['肝细胞癌', '肝癌', '肝占位', '肝肿瘤']),
        ('肝囊肿', ['肝囊肿', '囊肿']),
        ('脾大', ['脾大', '脾脏肿大', '脾肿大']),
        ('食管胃底静脉曲张', ['食管胃底静脉曲张']),
        ('脾功能亢进', ['脾功能亢进', '脾亢']),
        ('胆管扩张', ['胆管扩张', '胆总管扩张']),
        ('胆囊结石', ['胆囊结石', '胆结石']),
        ('肾结石', ['肾结石']),
        ('输尿管结石', ['输尿管结石']),
        ('膀胱结石', ['膀胱结石']),
        ('前列腺增生', ['前列腺增生', '前列腺肥大']),
        ('子宫肌瘤', ['子宫肌瘤', '子宫肿瘤']),
        ('卵巢囊肿', ['卵巢囊肿']),
        ('甲状腺结节', ['甲状腺结节', '甲状腺占位']),
        ('淋巴结肿大', ['淋巴结肿大', '肿大淋巴结']),
        ('动脉硬化', ['动脉硬化', '动脉粥样硬化']),
        ('冠心病', ['冠心病', '冠状动脉粥样硬化']),
        ('脑梗死', ['脑梗死', '脑梗', '缺血性脑卒中']),
        ('脑出血', ['脑出血', '出血性脑卒中']),
        ('脑积水', ['脑积水']),
        ('鼻窦炎', ['鼻窦炎']),
        ('中耳炎', ['中耳炎']),
        ('鼻咽癌', ['鼻咽癌', '鼻咽部占位']),
        ('喉癌', ['喉癌', '喉部占位']),
        ('肺炎', ['肺炎', '肺部感染']),
        ('肺结核', ['肺结核', '结核']),
        ('肺脓肿', ['肺脓肿', '脓肿']),
        ('胸腔积液', ['胸腔积液', '胸水']),
        ('积液', ['积液', '渗出液','肋膈角变钝','少量积液']),
        ('气胸', ['气胸','透亮区','无肺纹理','肺组织压缩','压缩肺']),
        ('纵隔肿瘤', ['纵隔肿瘤', '纵隔占位']),
        ('心包炎', ['心包炎']),
        ('心肌病', ['心肌病']),
        ('主动脉夹层', ['主动脉夹层', '夹层动脉瘤']),
        ('动脉瘤', ['动脉瘤']),
        ('阑尾炎', ['阑尾炎', '阑尾炎症']),
        ('肿瘤复发', ['肿瘤复发', '复发', '复发转移']),
        ('肿瘤转移', ['肿瘤转移', '转移', '转移灶']),
        ('肝右后叶', ['肝右后叶', '肝右叶', '肝左叶', '肝右前叶', '肝左外叶', '肝左内叶']),
        ('克罗恩病', ['克罗恩病', '克隆病']),
        ('便秘', ['便秘', '粪便潴留']),
        ('肠梗阻', ['肠梗阻', '机械性肠梗阻']),
        ('腹主动脉', ['腹主动脉', '主动脉']),
        ('血管分支', ['血管分支', '分支血管']),
        ('真腔', ['真腔', '假腔']),
        ('溃疡性结肠炎', ['溃疡性结肠炎', '结肠炎']),
        ('神经源性肿瘤', ['神经源性肿瘤', '神经鞘瘤', '神经纤维瘤']),
        ('肉瘤', ['肉瘤', '恶性肿瘤']),
        ('腹膜后肿瘤', ['腹膜后肿瘤', '腹膜后占位']),
        ('穿刺活检', ['穿刺活检', '活检']),
        ('活动期', ['活动期', '急性发作']),
        ('累及', ['累及', '侵犯', '侵犯范围']),
        ('颅内出血', ['颅内出血', '脑出血', '蛛网膜下腔出血']),
        ('头皮血肿', ['头皮血肿', '皮下血肿']),
        ('线状骨折', ['线状骨折', '裂缝骨折', '骨裂']),
        ('占位性病变', ['占位性病变', '占位']),
        ('急性出血', ['急性出血', '新鲜出血']),
        ('脑梗死', ['脑梗死', '脑梗', '缺血性脑卒中', '陈旧性脑梗死']),
        ('脑萎缩', ['脑萎缩', '老年性脑萎缩', '皮质萎缩']),
        ('垂体大腺瘤', ['垂体大腺瘤', '垂体腺瘤', '垂体瘤']),
        ('垂体MRI', ['垂体MRI', 'MRI检查']),
        ('末段回肠', ['末段回肠', '回肠末端']),
        ('脑积水', ['脑积水', '梗阻性脑积水']),
        ('多发转移瘤', ['多发转移瘤', '转移瘤', '脑转移']),
        ('脑脓肿', ['脑脓肿', '脓肿']),
        ('神经外科', ['神经外科', '外科就诊']),
        ('抗感染治疗', ['抗感染治疗', '抗炎治疗']),
        ('手术引流', ['手术引流', '引流术']),
        ('脱髓鞘改变', ['脱髓鞘改变', '脱髓鞘', '白质脱髓鞘']),
        ('弥漫性脑萎缩', ['弥漫性脑萎缩', '全脑萎缩']),
        ('阿尔茨海默病', ['阿尔茨海默病', '老年痴呆']),
        ('治疗方案', ['治疗方案', '调整治疗']),
        ('影像学表现', ['影像学表现', '影像表现']),
        ('较前进展', ['较前进展', '进展', '加重']),
        ('较前略有进展', ['较前略有进展', '略有进展']),
        ('硬膜下血肿', ['硬膜下血肿', '慢性硬膜下血肿', '硬膜下出血']),
        ('吸收期', ['吸收期', '恢复期', '愈合期']),
        ('缺氧缺血性脑病', ['缺氧缺血性脑病', 'HIE']),
        ('随访观察', ['随访观察', '继续随访']),
        ('定期复查', ['定期复查', '复查']),
        ('新生儿', ['新生儿', '婴儿']),
        ('脑囊虫病', ['脑囊虫病', '囊虫病']),
        ('钙化期', ['钙化期', '慢性期']),
        ('脑多发钙化', ['脑多发钙化', '多发钙化灶']),
        ('积极治疗', ['积极治疗', '对症治疗']),
        ('结合临床', ['结合临床', '结合病史']),
        ('肝硬化', ['肝硬化', '肝硬变']),
        ('胆囊炎', ['胆囊炎', '胆囊炎症']),
        ('形态正常', ['形态正常', '形态规整']),
        ('密度均匀', ['密度均匀', '实质密度均匀']),
        ('未见异常', ['未见异常', '未见明显异常', '无异常']),
        ('未见病变', ['未见病变', '未发现病变']),
        ('退行性关节炎', ['退行性关节炎', '退行性关节病', '退行性变']),
    ]

    QUALIFIER_PATTERNS = [
        '明确', '明显', '轻微', '少许', '少量', '大量',
        '多个', '单个', '少许', '散在', '局部', '广泛',
        '可疑', '可能', '疑似', '考虑', '提示', '倾向于',
        '约', '大小约', '直径约', '范围约',
        '首先考虑', '不能完全排除', '不除外', '可能与',
        '初步考虑', '可能性大', '可能性小',
        '以...为主', '倾向于', '不典型',
        '高度可疑', '高度怀疑',
    ]

    def __init__(self):
        self.logger = get_logger('FindingParser')
        self.negation_re = re.compile('|'.join(self.NEGATION_PATTERNS))
        self.qualifier_re = re.compile('|'.join(self.QUALIFIER_PATTERNS))

    def _detect_negation(self, text: str) -> bool:
        return bool(self.negation_re.search(text))

    def _detect_laterality(self, text: str) -> Optional[str]:
        for lat, patterns in self.LATERALITY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    return lat
        return None

    def _extract_anatomical_site(self, text: str) -> Optional[str]:
        for site in self.ANATOMICAL_SITES:
            if site in text:
                return site

        for pattern in self.BONE_SPECIFIC_PATTERNS:
            if re.search(pattern, text):
                return pattern

        bone_part_match = re.search(
            r'(股骨|胫骨|腓骨|肱骨|尺骨|桡骨|髌骨)(颈|头|体|髁|踝|幹|结节|平台)?',
            text
        )
        if bone_part_match:
            return bone_part_match.group()

        fracture_match = re.search(
            r'(股骨颈|股骨头|胫骨平台|肱骨头|锁骨|肋骨|椎体)(?:骨折)',
            text
        )
        if fracture_match:
            return fracture_match.group(1)

        return None

    def _extract_lesion_type(self, text: str) -> Optional[str]:
        all_patterns = []
        for lesion_name, patterns in self.LESION_TYPES:
            for pattern in patterns:
                all_patterns.append((pattern, lesion_name))

        all_patterns.sort(key=lambda x: len(x[0]), reverse=True)

        for pattern, lesion_name in all_patterns:
            if pattern in text:
                return lesion_name

        return None

    def _extract_qualifier(self, text: str) -> Optional[str]:
        matches = self.qualifier_re.findall(text)
        return ''.join(matches) if matches else None

    def _split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'[。；;！!？?]', text)
        return [s.strip() for s in sentences if s.strip()]

    def parse(self, text: str) -> List[Finding]:
        findings = []
        sentences = self._split_sentences(text)

        self.logger.debug(f"开始解析文本: {text[:50]}...")
        self.logger.debug(f"分割为 {len(sentences)} 个句子")

        for sentence in sentences:
            if len(sentence) < 3:
                continue

            sub_segments = self._split_by_comma(sentence)

            if len(sub_segments) > 1:
                for segment in sub_segments:
                    if len(segment) < 3:
                        continue
                    findings.extend(self._parse_segment(segment))
            else:
                findings.extend(self._parse_segment(sentence))

        self.logger.debug(f"解析完成，提取到 {len(findings)} 个发现项")
        return findings

    def _split_by_comma(self, text: str) -> List[str]:
        segments = re.split(r'[,，]', text)
        result = []

        for i, seg in enumerate(segments):
            seg = seg.strip()
            if not seg:
                continue

            prev_seg = segments[i-1].strip() if i > 0 else ""

            if self._extract_anatomical_site(seg) and self._extract_lesion_type(seg):
                result.append(seg)
            elif prev_seg and not self._extract_lesion_type(prev_seg):
                result[-1] = result[-1] + '，' + seg
            else:
                result.append(seg)

        return result

    def _parse_segment(self, segment: str) -> List[Finding]:
        findings = []
        negative_descriptors = ['形态正常', '密度均匀', '未见异常', '未见病变', '正常']
        is_negative_descriptor = any(d in segment for d in negative_descriptors)

        has_negation = self._detect_negation(segment)
        laterality = self._detect_laterality(segment)
        site = self._extract_anatomical_site(segment)
        lesion = self._extract_lesion_type(segment)
        qualifier = self._extract_qualifier(segment)

        if is_negative_descriptor:
            polarity = False
        elif has_negation and lesion:
            sign_negations = ['无肺纹理', '透亮区', '无纹理']
            is_sign_negation = any(s in segment for s in sign_negations)
            if is_sign_negation:
                polarity = True
            else:
                polarity = False
        elif has_negation:
            polarity = False
        else:
            polarity = True

        if site and lesion:
            confidence = self._calculate_confidence(
                segment, site, lesion, laterality
            )
            findings.append(Finding(
                anatomical_site=site,
                lesion_type=lesion,
                polarity=polarity,
                qualifier=qualifier,
                laterality=laterality,
                raw_text=segment,
                confidence=confidence
            ))
        elif site and polarity == False and '异常' in segment:
            confidence = self._calculate_confidence(
                segment, site, '异常', laterality
            )
            findings.append(Finding(
                anatomical_site=site,
                lesion_type='异常',
                polarity=polarity,
                qualifier=qualifier,
                laterality=laterality,
                raw_text=segment,
                confidence=confidence * 0.9
            ))
        elif polarity == False and '异常' in segment:
            confidence = self._calculate_confidence(
                segment, '全身', '异常', laterality
            )
            findings.append(Finding(
                anatomical_site='全身',
                lesion_type='异常',
                polarity=polarity,
                qualifier=qualifier,
                laterality=laterality,
                raw_text=segment,
                confidence=confidence * 0.85
            ))

        return findings

    UNCERTAINTY_QUALIFIERS = ['可疑', '可能', '疑似', '考虑', '提示', '倾向于', '轻微', '少许', '少量', '轻度']

    def _calculate_confidence(self, text: str, site: str, lesion: str,
                             laterality: Optional[str]) -> float:
        score = 0.7

        if site in text and lesion in text:
            score += 0.2

        if laterality:
            score += 0.05

        if len(text) >= 5:
            score += 0.05

        for qual in self.UNCERTAINTY_QUALIFIERS:
            if qual in text:
                score -= 0.2
                break

        return max(0.3, min(1.0, score))
