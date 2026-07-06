import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radiology_checker.core.main import ContradictionDetector, format_result


test_cases = [
    {
        "name": "正常报告-无矛盾",
        "findings": "颅骨骨板完整。脑实质结构清晰，未见异常密度影。脑室系统形态大小正常，脑沟及脑池无增宽，中线结构无移位。",
        "conclusion": "颅脑CT扫描未见异常。",
        "expected": "无矛盾"
    },
    {
        "name": "矛盾-有病变vs无异常",
        "findings": "左肺上叶见一约0.5cm结节影，边界清晰，密度均匀。",
        "conclusion": "双肺未见异常。",
        "expected": "矛盾"
    },
    {
        "name": "矛盾-左侧vs右侧",
        "findings": "右侧胸腔可见少量积液。",
        "conclusion": "左侧胸腔积液。",
        "expected": "矛盾"
    },
    {
        "name": "矛盾-骨折vs无骨折",
        "findings": "左桡骨远端见透亮线影，骨质不连续，断端对位对线尚可。",
        "conclusion": "左桡骨远端未见骨折。",
        "expected": "矛盾"
    },
    {
        "name": "正常报告-有病变描述一致",
        "findings": "颈椎生理曲度变直，颈3/4、4/5椎间盘向椎体后缘轻度突出，相应硬膜囊略受压。",
        "conclusion": "颈椎退行性改变，颈3/4、4/5椎间盘轻度突出。",
        "expected": "无矛盾"
    },
    {
        "name": "矛盾-增生vs无增生",
        "findings": "腰椎椎体边缘见骨质增生。",
        "conclusion": "腰椎未见骨质增生。",
        "expected": "矛盾"
    },
    {
        "name": "正常报告-详细描述",
        "findings": "肝脏形态大小正常，肝实质密度均匀，未见异常密度影，肝内外胆管无扩张。胆囊不大，壁不厚，内未见异常密度影。脾脏形态大小正常，密度均匀。",
        "conclusion": "上腹部CT扫描未见明显异常。",
        "expected": "无矛盾"
    },
    {
        "name": "矛盾-肿块vs无肿块",
        "findings": "右乳腺外上象限见一约2.0cm×1.5cm肿块影，边界不清，密度不均匀。",
        "conclusion": "右乳腺未见肿块。",
        "expected": "矛盾"
    }
]


def main():
    print("=" * 80)
    print("新训练模型测试")
    print("=" * 80)
    
    detector = ContradictionDetector(use_neuro_model=True)
    
    correct = 0
    total = len(test_cases)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n--- 测试案例 {i}: {case['name']} ---")
        print(f"所见: {case['findings']}")
        print(f"结论: {case['conclusion']}")
        print(f"预期: {case['expected']}")
        
        result = detector.analyze(case['findings'], case['conclusion'])
        
        if result['is_conflict']:
            actual = "矛盾"
            print(f"检测结果: 🔴 {actual}")
        else:
            actual = "无矛盾"
            print(f"检测结果: 🟢 {actual}")
        
        print(f"判定来源: {result['decision_source']}")
        print(f"置信度: {result['final_confidence']:.2f}")
        if result['nli_used']:
            print(f"NLI解释: {result['nli_explanation']}")
        
        if actual == case['expected']:
            correct += 1
            print("✅ 正确")
        else:
            print("❌ 错误")
    
    print("\n" + "=" * 80)
    print(f"测试结果: {correct}/{total} 正确")
    print(f"准确率: {(correct/total)*100:.2f}%")
    print("=" * 80)


if __name__ == '__main__':
    main()