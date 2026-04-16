"""
Train unweighted and JMDS models on the full training split, gather per-epoch
eval loss from seed log files, and dump predictions + learning curves to
viz_data.json. Intended to be run after all seed-level training jobs have
finished (see run_seed.py).
"""
import os, json, re, glob
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F
from dataclasses import dataclass
from typing import Any
from datasets import Dataset, DatasetDict
from transformers import (AutoTokenizer, AutoModelForTokenClassification,
    DataCollatorForTokenClassification, TrainingArguments, Trainer, set_seed)

LABELS   = ["O","PERSON","GOD","LOC","NORP","LANGUAGE"]
label2id = {l:i for i,l in enumerate(LABELS)}
id2label = {i:l for l,i in label2id.items()}

with open('./data/combined_confidence.json', encoding='utf-8') as f:
    train_data = json.load(f)
with open('./data/test_data.json', encoding='utf-8') as f:
    test_data = json.load(f)

model_name = "xlm-roberta-base"
tokenizer  = AutoTokenizer.from_pretrained(model_name, use_fast=True)

def tok_weighted(batch):
    tok = tokenizer(batch['tokens'], is_split_into_words=True, truncation=True)
    all_labels, all_weights = [], []
    for i in range(len(batch['tokens'])):
        wids = tok.word_ids(batch_index=i)
        li, wi = [], []
        for wid in wids:
            if wid is None:
                li.append(-100); wi.append(0.0)
            else:
                li.append(label2id[batch['ner_tags'][i][wid]])
                wi.append(float(batch['confidences'][i][wid]))
        all_labels.append(li); all_weights.append(wi)
    tok['labels'] = all_labels; tok['weights'] = all_weights
    return tok

def tok_uniform(batch):
    tok = tokenizer(batch['tokens'], is_split_into_words=True, truncation=True)
    all_labels, all_weights = [], []
    for i in range(len(batch['tokens'])):
        wids = tok.word_ids(batch_index=i)
        li, wi = [], []
        for wid in wids:
            if wid is None:
                li.append(-100); wi.append(0.0)
            else:
                li.append(label2id[batch['ner_tags'][i][wid]]); wi.append(1.0)
        all_labels.append(li); all_weights.append(wi)
    tok['labels'] = all_labels; tok['weights'] = all_weights
    return tok

@dataclass
class WeightedCollator:
    tokenizer: Any
    def __call__(self, features):
        weights = [f.pop('weights') for f in features]
        base  = DataCollatorForTokenClassification(tokenizer=self.tokenizer, padding=True)
        batch = base(features)
        seq_len = batch['input_ids'].shape[1]
        padded = [list(w)[:seq_len] + [0.0]*(seq_len - len(w)) for w in weights]
        batch['weights'] = torch.tensor(padded, dtype=torch.float)
        return batch

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop('labels')
        weights = inputs.pop('weights')
        out     = model(**inputs)
        logits  = out.logits
        B, L, K = logits.shape
        logits  = logits.view(B*L, K)
        labels  = labels.view(B*L)
        weights = weights.view(B*L).to(logits.device)
        mask    = labels != -100
        loss    = (F.cross_entropy(logits[mask], labels[mask], reduction='none') * weights[mask]).sum() / (weights[mask].sum() + 1e-8)
        return (loss, out) if return_outputs else loss

class UnweightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs.pop('weights', None)
        labels = inputs.pop('labels')
        out    = model(**inputs)
        logits = out.logits
        B, L, K = logits.shape
        logits  = logits.view(B*L, K)
        labels  = labels.view(B*L)
        mask    = labels != -100
        loss    = F.cross_entropy(logits[mask], labels[mask])
        return (loss, out) if return_outputs else loss

collator = WeightedCollator(tokenizer)
ds = DatasetDict({
    'train': Dataset.from_list(train_data),
    'test':  Dataset.from_list(test_data),
})

results = {}
for method, TrainCls, tok_fn in [
        ('unweighted', UnweightedTrainer, tok_uniform),
        ('jmds',       WeightedTrainer,   tok_weighted)]:
    set_seed(42)
    print(f"\n=== Training {method} ===")
    tok_train = ds['train'].map(tok_fn,      batched=True, remove_columns=ds['train'].column_names)
    tok_test  = ds['test'].map(tok_uniform,  batched=True, remove_columns=ds['test'].column_names)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=len(LABELS), id2label=id2label, label2id=label2id)
    trainer = TrainCls(
        model=model,
        args=TrainingArguments(
            output_dir=f'eval_{method}', learning_rate=2e-5,
            per_device_train_batch_size=8, per_device_eval_batch_size=16,
            num_train_epochs=3, eval_strategy='epoch', save_strategy='no',
            report_to='none', fp16=torch.cuda.is_available(),
            remove_unused_columns=False, seed=42),
        train_dataset=tok_train, eval_dataset=tok_test, data_collator=collator)
    trainer.train()
    pred = trainer.predict(tok_test)
    preds_flat, labels_flat = [], []
    for p_seq, l_seq in zip(np.argmax(pred.predictions, axis=-1), pred.label_ids):
        for p, l in zip(p_seq, l_seq):
            if l == -100: continue
            preds_flat.append(id2label[int(p)])
            labels_flat.append(id2label[int(l)])
    results[method] = {'preds': preds_flat, 'labels': labels_flat}
    print(f'{method}: {len(preds_flat)} tokens predicted')

# Collect learning curves from all 10 seed log files
logs = {}
for logfile in glob.glob('./log_*_seed*.log'):
    name = logfile.replace('./log_', '').replace('.log', '')
    losses = []
    with open(logfile) as f:
        for line in f:
            m = re.search(r"'eval_loss': '([0-9.]+)'.*'epoch': '([0-9.]+)'", line)
            if m:
                losses.append({'epoch': float(m.group(2)), 'eval_loss': float(m.group(1))})
    if losses:
        logs[name] = losses

print(f'Learning curves collected: {len(logs)} runs')
with open('./viz_data.json', 'w') as f:
    json.dump({'predictions': results, 'learning_curves': logs}, f)
print('Done. Saved viz_data.json')
