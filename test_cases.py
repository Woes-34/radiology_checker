TEST_CASES = [
    {
        "name": "正常-颅脑CT无异常",
        "findings": "颅骨骨板完整。脑实质结构清晰，未见异常密度影。脑室系统形态大小正常，脑沟及脑池无增宽，中线结构无移位。",
        "conclusion": "颅脑CT扫描未见异常。",
        "expected": "no_conflict"
    },
    {
        "name": "矛盾-结节vs无异常",
        "findings": "左肺上叶见一约0.5cm结节影，边界清晰，密度均匀。",
        "conclusion": "双肺未见异常。",
        "expected": "conflict"
    },
    {
        "name": "矛盾-右侧vs左侧胸腔积液",
        "findings": "右侧胸腔可见少量积液。",
        "conclusion": "左侧胸腔积液。",
        "expected": "conflict"
    },
    {
        "name": "矛盾-骨折vs无骨折",
        "findings": "左桡骨远端见透亮线影，骨质不连续，断端对位对线尚可。",
        "conclusion": "左桡骨远端未见骨折。",
        "expected": "conflict"
    },
    {
        "name": "正常-椎间盘突出描述一致",
        "findings": "颈椎生理曲度变直，颈3/4、4/5椎间盘向椎体后缘轻度突出，相应硬膜囊略受压。",
        "conclusion": "颈椎退行性改变，颈3/4、4/5椎间盘轻度突出。",
        "expected": "no_conflict"
    },
    {
        "name": "矛盾-增生vs无增生",
        "findings": "腰椎椎体边缘见骨质增生。",
        "conclusion": "腰椎未见骨质增生。",
        "expected": "conflict"
    },
    {
        "name": "正常-腹部CT无异常",
        "findings": "肝脏形态大小正常，肝实质密度均匀，未见异常密度影，肝内外胆管无扩张。胆囊不大，壁不厚，内未见异常密度影。脾脏形态大小正常，密度均匀。",
        "conclusion": "上腹部CT扫描未见明显异常。",
        "expected": "no_conflict"
    },
    {
        "name": "矛盾-肿块vs无肿块",
        "findings": "右乳腺外上象限见一约2.0cm×1.5cm肿块影，边界不清，密度不均匀。",
        "conclusion": "右乳腺未见肿块。",
        "expected": "conflict"
    },
    {
        "name": "矛盾-炎症vs无炎症",
        "findings": "右肺下叶见斑片状密度增高影，边界模糊，考虑炎症。",
        "conclusion": "双肺未见炎症。",
        "expected": "conflict"
    },
    {
        "name": "矛盾-积液vs无积液",
        "findings": "心包腔内可见少量积液。",
        "conclusion": "心包未见积液。",
        "expected": "conflict"
    },
    {
        "name": "矛盾-狭窄vs无狭窄",
        "findings": "冠状动脉左前降支中段见局限性狭窄，约50%。",
        "conclusion": "冠状动脉未见狭窄。",
        "expected": "conflict"
    },
    {
        "name": "矛盾-扩张vs无扩张",
        "findings": "肝内外胆管明显扩张。",
        "conclusion": "肝内外胆管无扩张。",
        "expected": "conflict"
    },
    {
        "name": "矛盾-肿大vs无肿大",
        "findings": "双侧颈部淋巴结肿大。",
        "conclusion": "颈部淋巴结未见肿大。",
        "expected": "conflict"
    },
    {
        "name": "矛盾-移位vs无移位",
        "findings": "右肱骨头向后移位。",
        "conclusion": "右肱骨头未见移位。",
        "expected": "conflict"
    },
    {
        "name": "矛盾-上叶vs下叶",
        "findings": "右肺上叶见结节影。",
        "conclusion": "右肺下叶结节。",
        "expected": "conflict"
    },
    {
        "name": "矛盾-左侧肾vs右侧肾",
        "findings": "左肾见一约1.0cm囊肿。",
        "conclusion": "右肾囊肿。",
        "expected": "conflict"
    },
    {
        "name": "正常-骨折愈合描述一致",
        "findings": "右胫骨骨折内固定术后，骨折线模糊，骨痂形成。",
        "conclusion": "右胫骨骨折术后改变，骨折愈合中。",
        "expected": "no_conflict"
    },
    {
        "name": "正常-肺纹理增多",
        "findings": "双肺纹理增多、紊乱，未见明显异常密度影。",
        "conclusion": "双肺纹理增多，未见明显异常。",
        "expected": "no_conflict"
    },
    {
        "name": "矛盾-骨质破坏vs无骨质破坏",
        "findings": "右股骨近端见骨质破坏区。",
        "conclusion": "右股骨近端未见骨质破坏。",
        "expected": "conflict"
    },
    {
        "name": "矛盾-钙化vs无钙化",
        "findings": "主动脉壁见钙化影。",
        "conclusion": "主动脉壁未见钙化。",
        "expected": "conflict"
    },
    {
        "name": "正常-肾结石",
        "findings": "左肾下盏见一约0.5cm高密度影，考虑结石。",
        "conclusion": "左肾结石。",
        "expected": "no_conflict"
    },
    {
        "name": "矛盾-脑梗死vs无梗死",
        "findings": "右侧基底节区见片状低密度影，边界不清，考虑脑梗死。",
        "conclusion": "头颅CT未见脑梗死。",
        "expected": "conflict"
    },
    {
        "name": "矛盾-出血vs无出血",
        "findings": "左侧丘脑见高密度影，考虑出血。",
        "conclusion": "颅内未见出血。",
        "expected": "conflict"
    },
    {
        "name": "正常-肝硬化",
        "findings": "肝脏体积缩小，表面凹凸不平，肝实质密度不均匀，肝裂增宽，考虑肝硬化。",
        "conclusion": "肝硬化。",
        "expected": "no_conflict"
    },
    {
        "name": "矛盾-气胸vs无气胸",
        "findings": "右侧胸腔见透亮区，无肺纹理，肺组织压缩约30%，考虑气胸。",
        "conclusion": "双侧胸腔未见气胸。",
        "expected": "conflict"
    },
]