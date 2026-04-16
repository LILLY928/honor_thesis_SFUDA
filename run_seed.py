"""
Single training run for one seed and one method.
Usage: python run_seed.py --seed 42 --method [unweighted|jmds]
"""
import os, sys, json, argparse
os.chdir(os.path.dirname(os.path.abspath(__file__)))

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

# Command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--seed",   type=int,  required=True)
parser.add_argument("--method", type=str,  required=True, choices=["unweighted", "jmds"])
args = parser.parse_args()

set_seed(args.seed)
print(f"Seed: {args.seed}  Method: {args.method}")

# Load data
with open("./data/combined_confidence.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)
with open("./data/test_data.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

print(f"Train: {len(train_data)}  Test: {len(test_data)}")

# Label set
LABELS   = ["O", "PERSON", "GOD", "LOC", "NORP", "LANGUAGE"]
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}

def _check(data, name):
    for ex in data:
        if len(ex["tokens"]) != len(ex["ner_tags"]):
            raise ValueError(f"{name}: length mismatch")
        bad = [t for t in ex["ner_tags"] if t not in label2id]
        if bad:
            raise ValueError(f"{name}: unknown labels {set(bad)}")

_check(train_data, "train")
_check(test_data,  "test")

ds = DatasetDict({
    "train": Dataset.from_list(train_data),
    "test":  Dataset.from_list(test_data),
})

model_name = "xlm-roberta-base"
tokenizer  = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# Tokenization functions
def tokenize_unweighted(batch):
    """Pseudo-labels, uniform weight = 1.0 for every token."""
    tok = tokenizer(batch["tokens"], is_split_into_words=True, truncation=True)
    all_labels, all_weights = [], []
    for i in range(len(batch["tokens"])):
        word_ids  = tok.word_ids(batch_index=i)
        labels_i  = batch["ner_tags"][i]
        label_ids, weight_ids = [], []
        for wid in word_ids:
            if wid is None:
                label_ids.append(-100)
                weight_ids.append(0.0)
            else:
                label_ids.append(label2id[labels_i[wid]])
                weight_ids.append(1.0)           # <-- uniform weight
        all_labels.append(label_ids)
        all_weights.append(weight_ids)
    tok["labels"]  = all_labels
    tok["weights"] = all_weights
    return tok

def tokenize_jmds(batch):
    """Pseudo-labels, JMDS confidence score as weight."""
    tok = tokenizer(batch["tokens"], is_split_into_words=True, truncation=True)
    all_labels, all_weights = [], []
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
                weight_ids.append(float(confs_i[wid]))  # <-- JMDS weight
        all_labels.append(label_ids)
        all_weights.append(weight_ids)
    tok["labels"]  = all_labels
    tok["weights"] = all_weights
    return tok

def tokenize_eval(batch):
    """Test set: uniform weight = 1.0."""
    tok = tokenizer(batch["tokens"], is_split_into_words=True, truncation=True)
    all_labels, all_weights = [], []
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
        all_labels.append(label_ids)
        all_weights.append(weight_ids)
    tok["labels"]  = all_labels
    tok["weights"] = all_weights
    return tok

raw_cols_train = ds["train"].column_names
raw_cols_test  = ds["test"].column_names

train_fn = tokenize_unweighted if args.method == "unweighted" else tokenize_jmds

tokenized_train = ds["train"].map(train_fn,      batched=True, remove_columns=raw_cols_train)
tokenized_test  = ds["test"].map(tokenize_eval,  batched=True, remove_columns=raw_cols_test)

# Data collator that pads the per-token weight vector alongside the batch
@dataclass
class WeightedCollator:
    tokenizer: Any
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        weights = [f.pop("weights") for f in features]
        base    = DataCollatorForTokenClassification(tokenizer=self.tokenizer, padding=True)
        batch   = base(features)
        seq_len = batch["input_ids"].shape[1]
        padded  = []
        for w in weights:
            w = list(w)[:seq_len] + [0.0] * (seq_len - len(w))
            padded.append(w)
        batch["weights"] = torch.tensor(padded, dtype=torch.float)
        return batch

collator = WeightedCollator(tokenizer)

# Trainer with weighted cross-entropy loss
class WeightedTrainer(Trainer):
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

# Model and training arguments
model = AutoModelForTokenClassification.from_pretrained(
    model_name, num_labels=len(LABELS), id2label=id2label, label2id=label2id,
)

save_dir = f"./models/{args.method}_seed{args.seed}"
os.makedirs(save_dir, exist_ok=True)

args_hf = TrainingArguments(
    output_dir=save_dir,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="no",
    report_to="none",
    fp16=torch.cuda.is_available(),
    remove_unused_columns=False,
    seed=args.seed,
)

TrainerClass = WeightedTrainer if args.method == "jmds" else Trainer

trainer_kwargs = dict(
    model=model,
    args=args_hf,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    data_collator=collator,
)

# Unweighted still needs the collator to handle weights column,
# but we override compute_loss to ignore it
if args.method == "unweighted":
    class UnweightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            inputs.pop("weights", None)   # discard weights, uniform loss
            labels  = inputs.pop("labels")
            outputs = model(**inputs)
            logits  = outputs.logits
            B, L, K = logits.shape
            logits  = logits.view(B * L, K)
            labels  = labels.view(B * L)
            mask    = labels != -100
            loss    = F.cross_entropy(logits[mask], labels[mask])
            return (loss, outputs) if return_outputs else loss
    TrainerClass = UnweightedTrainer

trainer = TrainerClass(**trainer_kwargs)

# Train
print("Training...")
trainer.train()

# Evaluate
print("Evaluating...")
pred   = trainer.predict(tokenized_test)
logits = pred.predictions
labels = pred.label_ids

preds_arr = np.argmax(logits, axis=-1)

# Per-label accuracy
stats = defaultdict(lambda: {"correct": 0, "total": 0})
for p_seq, l_seq in zip(preds_arr, labels):
    for p, l in zip(p_seq, l_seq):
        if l == -100:
            continue
        lname = id2label[int(l)]
        stats[lname]["total"] += 1
        if p == l:
            stats[lname]["correct"] += 1

per_label = {k: v["correct"] / v["total"] for k, v in stats.items()}
macro     = sum(per_label.values()) / len(per_label)
token_acc = sum(v["correct"] for v in stats.values()) / sum(v["total"] for v in stats.values())

print(f"\n--- Results (seed={args.seed}, method={args.method}) ---")
for lbl, acc in per_label.items():
    print(f"  {lbl:10s}  acc={acc:.4f}  (n={stats[lbl]['total']})")
print(f"Token accuracy : {token_acc:.4f}")
print(f"Macro accuracy : {macro:.4f}")

# Save results
result = {
    "seed":      args.seed,
    "method":    args.method,
    "token_acc": round(token_acc, 6),
    "macro":     round(macro, 6),
    **{k: round(v, 6) for k, v in per_label.items()},
}
os.makedirs("./results", exist_ok=True)
out_path = f"./results/results_{args.method}_seed{args.seed}.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved: {out_path}")
