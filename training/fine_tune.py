import os
import sys
import json
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)


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


def compute_precision_recall(labels, preds, class_names):
    num_classes = len(class_names)
    result = {}
    
    for cls_idx, cls_name in enumerate(class_names):
        tp = ((preds == cls_idx) & (labels == cls_idx)).sum()
        fp = ((preds == cls_idx) & (labels != cls_idx)).sum()
        fn = ((preds != cls_idx) & (labels == cls_idx)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        result[f"{cls_name}_precision"] = precision
        result[f"{cls_name}_recall"] = recall
        result[f"{cls_name}_f1"] = f1
    
    return result


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    accuracy = compute_accuracy(labels, preds)
    f1_macro = compute_f1(labels, preds, average='macro')
    f1_weighted = compute_f1(labels, preds, average='weighted')
    
    class_names = ['contradiction', 'entailment', 'neutral']
    pr_results = compute_precision_recall(labels, preds, class_names)
    
    result = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
    }
    result.update(pr_results)
    
    return result


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


MEDICAL_BASE_MODELS = [
    'trueto/medbert-base-chinese',
    'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12',
    'radiology_checker/models/medbert-base-chinese',
]


def fine_tune_model(
    model_name: str = 'trueto/medbert-base-chinese',
    train_file: str = 'radiology_checker/data/train.jsonl',
    val_file: str = 'radiology_checker/data/val.jsonl',
    test_file: str = 'radiology_checker/data/test.jsonl',
    output_dir: str = 'radiology_checker/models/fine_tuned_nli',
    num_epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_length: int = 512
):
    print(f"加载模型: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if model_name in MEDICAL_BASE_MODELS:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3,
            ignore_mismatched_sizes=True
        )
        print(f"使用医学预训练模型，添加NLI分类头")
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f"使用设备: {device}")
    
    train_data = []
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                train_data.append(json.loads(line))
    
    val_data = []
    with open(val_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                val_data.append(json.loads(line))
    
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                test_data.append(json.loads(line))
    
    print(f"训练集: {len(train_data)} 条")
    print(f"验证集: {len(val_data)} 条")
    print(f"测试集: {len(test_data)} 条")
    
    train_dataset = NLIDataset(train_data, tokenizer, max_length)
    val_dataset = NLIDataset(val_data, tokenizer, max_length)
    test_dataset = NLIDataset(test_data, tokenizer, max_length)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=100,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model='f1_macro',
        greater_is_better=True,
        fp16=True,
        gradient_accumulation_steps=4,
        report_to='none',
        disable_tqdm=True,
        use_cpu=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    print("开始微调...")
    trainer.train()
    
    print("\n评估测试集...")
    test_results = trainer.predict(test_dataset)
    print("测试集结果:")
    for k, v in test_results.metrics.items():
        print(f"  {k}: {v:.4f}")
    
    print("\n保存模型...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"模型已保存到: {output_dir}")
    
    return test_results.metrics


def evaluate_model(
    model_path: str,
    test_file: str = 'radiology_checker/data/test.jsonl',
    max_length: int = 512
):
    print(f"加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                test_data.append(json.loads(line))
    
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
    
    print(f"准确率: {accuracy:.4f}")
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
    fine_tune_model(
        model_name='radiology_checker/models/medbert-base-chinese',
        train_file='radiology_checker/data/train.jsonl',
        val_file='radiology_checker/data/val.jsonl',
        test_file='radiology_checker/data/test.jsonl',
        output_dir='radiology_checker/models/fine_tuned_nli',
        num_epochs=3,
        batch_size=2,
        learning_rate=2e-5,
        max_length=256
    )
