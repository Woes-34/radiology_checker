import os
import sys
from typing import Optional, Dict, Any

# 使用CMedBERT模型（trueto/medbert-base-chinese）

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from radiology_checker.nli.interface import NLIModelInterface, NLIResult
from radiology_checker.logger import get_logger

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class NeuroNLIModel(NLIModelInterface):
    MEDICAL_BASE_MODELS = [
        'trueto/medbert-base-chinese',
        'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12',
    ]
    
    def __init__(self, model_name: str = None, device: str = 'cpu', use_fine_tuned: bool = True, use_medical_model: bool = True):
        self.logger = get_logger('NeuroNLIModel')
        self.device = device
        self.tokenizer = None
        self.model = None
        self.labels = ['contradiction', 'entailment', 'neutral']
        self.use_medical_model = use_medical_model
        
        if model_name:
            self.model_name = model_name
        elif use_fine_tuned:
            fine_tuned_auto_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'fine_tuned_auto')
            fine_tuned_nli_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'fine_tuned_nli')
            if os.path.exists(fine_tuned_auto_path):
                self.model_name = fine_tuned_auto_path
            elif os.path.exists(fine_tuned_nli_path):
                self.model_name = fine_tuned_nli_path
            elif use_medical_model:
                self.model_name = 'trueto/medbert-base-chinese'
            else:
                self.model_name = 'IDEA-CCNL/Erlangshen-Roberta-110M-NLI'
        elif use_medical_model:
            self.model_name = 'trueto/medbert-base-chinese'
        else:
            self.model_name = 'IDEA-CCNL/Erlangshen-Roberta-110M-NLI'

        if TRANSFORMERS_AVAILABLE:
            self._load_model()
    
    def _load_model(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.model_name in self.MEDICAL_BASE_MODELS:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    num_labels=3,
                    ignore_mismatched_sizes=True
                )
            else:
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            self.logger.info(f"神经NLI模型加载成功: {self.model_name}")
        except Exception as e:
            self.logger.error(f"加载模型 {self.model_name} 失败: {e}")
            fallback_models = []
            if self.use_medical_model:
                fallback_models.append('IDEA-CCNL/Erlangshen-Roberta-110M-NLI')
            for fallback_model in fallback_models:
                if self.model_name != fallback_model:
                    self.logger.info(f"尝试加载回退模型: {fallback_model}")
                    try:
                        self.model_name = fallback_model
                        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                        self.model.to(self.device)
                        self.model.eval()
                        self.logger.info(f"回退模型加载成功: {self.model_name}")
                        return
                    except Exception as e2:
                        self.logger.error(f"回退模型加载也失败: {e2}")
            self.tokenizer = None
            self.model = None

    def is_available(self) -> bool:
        return self.model is not None and TRANSFORMERS_AVAILABLE

    def predict(self, premise: str, hypothesis: str) -> NLIResult:
        if not self.is_available():
            return NLIResult(
                is_conflict=False,
                confidence=0.0,
                explanation="神经NLI模型不可用"
            )
        
        try:
            inputs = self.tokenizer(
                premise, 
                hypothesis, 
                return_tensors='pt', 
                padding=True, 
                truncation=True, 
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                
            contradiction_prob = probs[0][0].item()
            entailment_prob = probs[0][1].item()
            neutral_prob = probs[0][2].item()
            
            predicted_label = self.labels[probs.argmax().item()]
            
            if predicted_label == 'contradiction':
                is_conflict = True
                confidence = contradiction_prob
                explanation = f"NLI判定：矛盾（概率={contradiction_prob:.4f}）"
            elif predicted_label == 'entailment':
                is_conflict = False
                confidence = entailment_prob
                explanation = f"NLI判定：蕴含（概率={entailment_prob:.4f}）"
            else:
                is_conflict = False
                confidence = neutral_prob
                explanation = f"NLI判定：中立（概率={neutral_prob:.4f}）"
            
            return NLIResult(
                is_conflict=is_conflict,
                confidence=confidence,
                explanation=explanation,
                is_ambiguous=False
            )
        
        except Exception as e:
            return NLIResult(
                is_conflict=False,
                confidence=0.0,
                explanation=f"神经NLI模型预测失败: {e}"
            )

    def predict_with_probs(self, premise: str, hypothesis: str) -> Dict[str, float]:
        if not self.is_available():
            return {'contradiction': 0.0, 'entailment': 0.0, 'neutral': 0.0}
        
        try:
            inputs = self.tokenizer(
                premise, 
                hypothesis, 
                return_tensors='pt', 
                padding=True, 
                truncation=True, 
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                
            return {
                'contradiction': probs[0][0].item(),
                'entailment': probs[0][1].item(),
                'neutral': probs[0][2].item()
            }
        
        except Exception as e:
            return {'contradiction': 0.0, 'entailment': 0.0, 'neutral': 0.0}
