import os
import sys
import json
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import AutoTokenizer, AutoModelForSequenceClassification


def compute_accuracy(labels, preds):
    return (labels == preds).mean()


def compute_f1(labels, preds, average='macro'):
    num_classes = len(np.unique(labels))
    f1_scores = []
    
    for cls in range(num_classes):
        tp = ((preds == cls) & (labels == cls)).sum()
        fp = ((preds == cls) & (labels != cls)).sum()
        fn = ((preds != cls) & (labels == cls)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    
    if average == 'macro':
        return np.mean(f1_scores)
    elif average == 'weighted':
        weights = [(labels == cls).sum() for cls in range(num_classes)]
        weights = np.array(weights) / len(labels)
        return np.sum(f1_scores * weights)
    return f1_scores


class NLIDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = {'contradiction': 0, 'entailment': 1, 'neutral': 2}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        premise = item['premise']
        hypothesis = item['hypothesis']
        label = self.label_map[item['label']]
        
        encoding = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_overflowing_tokens=False,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def evaluate_model(model_path, test_file, max_length=512):
    print(f"加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    print(f"使用设备: {device}")
    
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                test_data.append(json.loads(line))
    
    print(f"测试集大小: {len(test_data)}")
    
    test_dataset = NLIDataset(test_data, tokenizer, max_length)
    
    predictions = []
    labels = []
    
    with torch.no_grad():
        for item in test_dataset:
            input_ids = item['input_ids'].unsqueeze(0).to(device)
            attention_mask = item['attention_mask'].unsqueeze(0).to(device)
            token_type_ids = item['token_type_ids'].unsqueeze(0).to(device)
            label = item['labels'].item()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            logits = outputs.logits
            pred = logits.argmax(dim=1).item()
            
            predictions.append(pred)
            labels.append(label)
    
    labels = np.array(labels)
    predictions = np.array(predictions)
    
    accuracy = compute_accuracy(labels, predictions)
    f1_macro = compute_f1(labels, predictions, average='macro')
    f1_weighted = compute_f1(labels, predictions, average='weighted')
    
    print(f"\n准确率: {accuracy:.4f}")
    print(f"F1 (macro): {f1_macro:.4f}")
    print(f"F1 (weighted): {f1_weighted:.4f}")
    
    class_names = ['contradiction', 'entailment', 'neutral']
    print("\n分类报告:")
    print(f"{'类别':<15} {'精确率':<8} {'召回率':<8} {'F1':<8}")
    print("-" * 40)
    
    for cls_idx, cls_name in enumerate(class_names):
        tp = ((predictions == cls_idx) & (labels == cls_idx)).sum()
        fp = ((predictions == cls_idx) & (labels != cls_idx)).sum()
        fn = ((predictions != cls_idx) & (labels == cls_idx)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{cls_name:<15} {precision:<8.4f} {recall:<8.4f} {f1:<8.4f}")


if __name__ == '__main__':
    evaluate_model('radiology_checker/models/fine_tuned_nli', 'radiology_checker/data/test.jsonl')
