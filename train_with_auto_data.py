import sys
import os
import json
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    accuracy = (labels == preds).mean()
    
    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels != 1)).sum()
    fn = ((preds != 1) & (labels == 1)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


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
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def main():
    print("加载数据...")
    train_data = load_data('radiology_checker/data/train_auto.jsonl')
    val_data = load_data('radiology_checker/data/val_auto.jsonl')
    test_data = load_data('radiology_checker/data/test_auto.jsonl')
    
    print(f"训练集: {len(train_data)}")
    print(f"验证集: {len(val_data)}")
    print(f"测试集: {len(test_data)}")
    
    print("\n加载模型...")
    tokenizer = AutoTokenizer.from_pretrained('radiology_checker/models/medbert-base-chinese')
    model = AutoModelForSequenceClassification.from_pretrained(
        'radiology_checker/models/medbert-base-chinese',
        num_labels=3,
        ignore_mismatched_sizes=True
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f"使用设备: {device}")
    
    train_dataset = NLIDataset(train_data, tokenizer, max_length=256)
    val_dataset = NLIDataset(val_data, tokenizer, max_length=256)
    test_dataset = NLIDataset(test_data, tokenizer, max_length=256)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    training_args = TrainingArguments(
        output_dir='radiology_checker/models/fine_tuned_auto',
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_dir='radiology_checker/logs/auto_training',
        logging_steps=500,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        fp16=True,
        report_to='none',
        disable_tqdm=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    print("\n开始训练...")
    trainer.train()
    
    print("\n评估测试集...")
    test_results = trainer.predict(test_dataset)
    print(f"测试集结果:")
    for k, v in test_results.metrics.items():
        print(f"  {k}: {v:.4f}")
    
    print("\n保存模型...")
    model.save_pretrained('radiology_checker/models/fine_tuned_auto')
    tokenizer.save_pretrained('radiology_checker/models/fine_tuned_auto')
    print("模型已保存到: radiology_checker/models/fine_tuned_auto")


if __name__ == '__main__':
    main()
