"""
Fine-tune XLM-RoBERTa-base on NEReus pseudo-labels with JMDS confidence weights.
Single-run version (seed=123). For multi-seed experiments, use run_seed.py.
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import json
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)

set_seed(123)

# Load data -------------------------------------------------------------------
with open("./data/combined_confidence.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)

with open("./data/test_data.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

print(f"Train samples: {len(train_data)}")
print(f"Test  samples: {len(test_data)}")

# Labels, tokenizer, dataset --------------------------------------------------
LABELS = ["O", "PERSON", "GOD", "LOC", "NORP", "LANGUAGE"]
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}

def _check(data, name):
    for ex in data:
        if len(ex["tokens"]) != len(ex["ner_tags"]):
            raise ValueError(f"{name}: length mismatch: {ex}")
        bad = [t for t in ex["ner_tags"] if t not in label2id]
        if bad:
            raise ValueError(f"{name}: unknown labels {set(bad)}")

_check(train_data, "train")
_check(test_data, "test")

ds = DatasetDict({
    "train": Dataset.from_list(train_data),
    "test":  Dataset.from_list(test_data),
})

model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# Tokenization with word-to-subword alignment -------------------------------
def tokenize_align_labels_and_weights(batch):
    tok = tokenizer(batch["tokens"], is_split_into_words=True, truncation=True)
    all_label_ids, all_weight_ids = [], []
    for i in range(len(batch["tokens"])):
        word_ids = tok.word_ids(batch_index=i)
        labels_i = batch["ner_tags"][i]
        confs_i  = batch["confidences"][i]
        label_ids, weight_ids = [], []
        for wid in word_ids:
            if wid is None:
                label_ids.append(-100)
                weight_ids.append(0.0)
            else:
                label_ids.append(label2id[labels_i[wid]])
                weight_ids.append(float(confs_i[wid]))
        all_label_ids.append(label_ids)
        all_weight_ids.append(weight_ids)
    tok["labels"]  = all_label_ids
    tok["weights"] = all_weight_ids
    return tok

def tokenize_and_align_labels_for_eval(batch):
    tok = tokenizer(batch["tokens"], is_split_into_words=True, truncation=True)
    all_label_ids, all_weight_ids = [], []
    for i in range(len(batch["tokens"])):
        word_ids = tok.word_ids(batch_index=i)
        labels_i = batch["ner_tags"][i]
        label_ids, weight_ids = [], []
        for wid in word_ids:
            if wid is None:
                label_ids.append(-100)
                weight_ids.append(0.0)
            else:
                label_ids.append(label2id[labels_i[wid]])
                weight_ids.append(1.0)
        all_label_ids.append(label_ids)
        all_weight_ids.append(weight_ids)
    tok["labels"]  = all_label_ids
    tok["weights"] = all_weight_ids
    return tok

raw_cols_train = ds["train"].column_names
raw_cols_test  = ds["test"].column_names

tokenized_train = ds["train"].map(tokenize_align_labels_and_weights,     batched=True, remove_columns=raw_cols_train)
tokenized_test  = ds["test"].map(tokenize_and_align_labels_for_eval,     batched=True, remove_columns=raw_cols_test)

@dataclass
class DataCollatorForTokenClassificationWithWeights:
    tokenizer: Any
    padding: bool = True
    max_length: int = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        weights = [f["weights"] for f in features]
        for f in features:
            f.pop("weights")
        base = DataCollatorForTokenClassification(tokenizer=self.tokenizer, padding=True, max_length=self.max_length)
        batch = base(features)
        seq_len = batch["input_ids"].shape[1]
        padded_weights = []
        for w in weights:
            w = w[:seq_len]
            w = w + [0.0] * (seq_len - len(w))
            padded_weights.append(w)
        batch["weights"] = torch.tensor(padded_weights, dtype=torch.float)
        return batch

collator = DataCollatorForTokenClassificationWithWeights(tokenizer)

class WeightedTokenTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        weights = inputs.pop("weights")
        outputs = model(**inputs)
        logits  = outputs.logits
        B, L, K = logits.shape
        logits  = logits.view(B * L, K)
        labels  = labels.view(B * L)
        weights = weights.view(B * L).to(logits.device)
        mask    = labels != -100
        logits  = logits[mask]
        labels  = labels[mask]
        weights = weights[mask]
        loss_per_token = F.cross_entropy(logits, labels, reduction="none")
        loss = (loss_per_token * weights).sum() / (weights.sum() + 1e-8)
        return (loss, outputs) if return_outputs else loss

# Train -----------------------------------------------------------------------
print("\nLoading model and starting training...")
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(LABELS),
    id2label=id2label,
    label2id=label2id,
)

args = TrainingArguments(
    output_dir="xlmr_weighted_ner",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    report_to="none",
    fp16=False,
    remove_unused_columns=False,
)

trainer = WeightedTokenTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    data_collator=collator,
)

trainer.train()

save_path = "./models/xlmr_weighted_ner_model"
os.makedirs(save_path, exist_ok=True)
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)
print("Saved model to:", save_path)

# Load saved model and predict ------------------------------------------------
print("\nLoading saved model for evaluation...")
model = AutoModelForTokenClassification.from_pretrained(save_path)
tokenizer = AutoTokenizer.from_pretrained(save_path)

eval_trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="eval_only",
        per_device_eval_batch_size=16,
        report_to="none",
        remove_unused_columns=False,
    ),
    eval_dataset=tokenized_test,
    data_collator=collator,
)

pred = eval_trainer.predict(tokenized_test)

# Accuracy --------------------------------------------------------------------
logits = pred.predictions
labels = pred.label_ids

def token_accuracy(logits, labels):
    preds = np.argmax(logits, axis=-1)
    correct = total = 0
    for p_seq, l_seq in zip(preds, labels):
        for p, l in zip(p_seq, l_seq):
            if l == -100:
                continue
            total += 1
            if p == l:
                correct += 1
    return correct / total

acc = token_accuracy(logits, labels)
print(f"\nToken-level accuracy: {acc:.4f}")

def per_label_accuracy(logits, labels, id2label):
    preds = np.argmax(logits, axis=-1)
    stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for p_seq, l_seq in zip(preds, labels):
        for p, l in zip(p_seq, l_seq):
            if l == -100:
                continue
            label = id2label[int(l)]
            stats[label]["total"] += 1
            if p == l:
                stats[label]["correct"] += 1
    print("\nPer-label accuracy:")
    for label, s in stats.items():
        a = s["correct"] / s["total"]
        print(f"  {label:10s}  acc={a:.4f}  (n={s['total']})")

per_label_accuracy(logits, labels, id2label)

def macro_accuracy(logits, labels, id2label):
    preds = np.argmax(logits, axis=-1)
    per_label = defaultdict(list)
    for p_seq, l_seq in zip(preds, labels):
        for p, l in zip(p_seq, l_seq):
            if l == -100:
                continue
            per_label[id2label[int(l)]].append(int(p == l))
    return sum(sum(v)/len(v) for v in per_label.values()) / len(per_label)

print(f"\nMacro accuracy: {macro_accuracy(logits, labels, id2label):.4f}")
