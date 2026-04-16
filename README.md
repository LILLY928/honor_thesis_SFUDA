# JMDS + FastText for Source-Free Domain Adaptation on Ancient Greek NER

Code and experiments for a joint model-data confidence (JMDS) framework that
weights pseudo-labels produced by a zero-shot LLM (GPT-4o-mini) with a
structural confidence signal derived from GMM clustering of FastText
embeddings, then fine-tunes XLM-RoBERTa-base on the NEReus Ancient Greek
corpus.

Entity classes: `O, PERSON, GOD, LOC, NORP` (with a residual `LANGUAGE`
folded into the label set for compatibility).

## Pipeline overview

```
nereus_dataset_cleaned.jsonl          (raw NEReus data)
        |
        v  (1) 80/20 sentence-level split
train_data.json / test_data.json
        |
        v  (2) GPT-4o-mini batch pseudo-labelling with log-probabilities
base_result.csv  -->  MPPL (exp of top log-prob)
        |
        v  (3) FastText embeddings + GMM (K=5) + temperature scaling (T=20)
fasttext_emb.json  -->  LPG / MINGAP
        |
        v  combined JMDS score = MPPL * LPG, clamped to [0.05, 1.0]
combined_confidence.json
        |
        v  (4) Weighted fine-tuning of XLM-RoBERTa-base
results_{method}_seed{k}.json  (one file per run)
        |
        v  aggregate + Wilcoxon signed-rank test
aggregated_results.json
```

## Repository layout

```
.
├── 1_train_test_split.ipynb              (1) build train/test splits
├── 2_gpt_pseudolabeling.ipynb            (2) OpenAI batch pseudo-labelling
├── 3_embedding+clustering.ipynb          (3) FastText + GMM + JMDS
├── 4_model_finetuning.ipynb              (4) XLM-R fine-tuning with JMDS weights
├── ablation_study.ipynb                  ablation runs
├── fine_tune_with_model_conf_only.ipynb  MPPL-only baseline
├── run_finetuning.py                     single-run training (seed=123)
├── run_seed.py                           per-seed CLI training script
├── aggregate.py                          aggregate per-seed results + Wilcoxon
├── collect_viz.py                        retrain + dump predictions/curves
├── collect_viz.slurm                     SLURM wrapper for collect_viz.py
├── generate_viz.py                       build thesis figures (a-e)
├── data/                                 input + intermediate data
├── results/                              per-seed result JSONs + aggregate
└── figures/                              generated PNG figures
```

## Data files

Files committed to the repo:

| File | Size | What it is |
|---|---|---|
| `data/nereus_dataset_cleaned.jsonl` | ~1.7 MB | Raw NEReus data after cleaning |
| `data/train_data.json` | ~3.2 MB | 80 % train split (true labels) |
| `data/test_data.json` | ~0.8 MB | 20 % test split (true labels) |
| `data/combined_confidence.json` | ~3.4 MB | Train split with pseudo-labels + JMDS weights |
| `data/word_to_probs.json` | ~0.8 MB | Word → LPG score (GMM-derived) |

Files **not** committed because they are too large or regenerable (see
`.gitignore`):

| File | Size | How to regenerate |
|---|---|---|
| `data/fasttext_emb.json` | ~56 MB | notebook 3, cell 5-7 (needs FastText model) |
| `data/base_requests.jsonl` | ~83 MB | notebook 2, cell 6 |
| `data/base_result.txt` | ~108 MB | download from OpenAI batch API (notebook 2) |
| `data/base_result.csv` | ~6 MB | notebook 2, cell 22 |

## External assets you must supply

1. **NEReus corpus.** `data/nereus_dataset_cleaned.jsonl` is a cleaned copy of
   the NEReus Ancient Greek NER dataset. Drop your own copy here if you want
   to re-run from step (1).
2. **Ancient Greek FastText model.** Notebook 3 expects a pre-trained skip-gram
   model at `models/grc_fasttext_skipgram_nn2_xn10_dim150.bin`
   (150-dim, character n-grams). Train your own or obtain one that matches
   these hyper-parameters.
3. **OpenAI API key.** Notebooks 2 and `ablation_study.ipynb` use the OpenAI
   Batch API for pseudo-labelling. Set `OPENAI_API_KEY` in your environment.

## Setup

Python 3.10+ recommended.

```bash
python -m venv venv
source venv/bin/activate       # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

If you plan to regenerate FastText embeddings, also install `fasttext`
(requires a C++ compiler):

```bash
pip install fasttext
```

## Reproducing the results

### 1. Train/test split

Open `1_train_test_split.ipynb` and run it top-to-bottom. It consumes
`data/nereus_dataset_cleaned.jsonl` and writes `data/train_data.json` and
`data/test_data.json`.

### 2. GPT-4o-mini pseudo-labelling (external teacher)

Open `2_gpt_pseudolabeling.ipynb`. The notebook:

1. Builds `data/base_requests.jsonl` from the training split (one request per
   token, sentence context included in the prompt).
2. Splits it into batches of ~50 000 requests and uploads each as an OpenAI
   Batch API job.
3. Polls the batches, downloads the outputs, parses the top-5 log-probabilities
   into `data/base_result.csv`.
4. Writes `data/train_data.json` style rows with pseudo-labels and the raw
   MPPL (= exp of the top log-prob) per token.

Set `OPENAI_API_KEY` before running. Each full run costs on the order of USD
tens depending on token count.

### 3. Embeddings, GMM, JMDS

Open `3_embedding+clustering.ipynb`. The notebook:

1. Loads `train_data.json` and the pre-trained Ancient Greek FastText model.
2. Writes `data/fasttext_emb.json` (word → 150-dim vector).
3. Fits a GMM with K=5 components, full covariance, `random_state=42`.
4. Applies temperature scaling with T=20 to the GMM log-responsibilities.
5. Computes MINGAP per word, normalises by the corpus maximum → LPG.
6. Writes `data/word_to_probs.json` (word → LPG).
7. Combines MPPL × LPG → JMDS, clamps to `[0.05, 1.0]`, and writes
   `data/combined_confidence.json`.

### 4. Fine-tune XLM-RoBERTa

Two options:

**Single run (quick check).** Run all cells of `4_model_finetuning.ipynb`, or
from the command line:

```bash
python run_finetuning.py
```

**Multi-seed experiment (main results).** Train 10 configurations
(2 methods × 5 seeds) with:

```bash
for method in unweighted jmds; do
  for seed in 42 123 456 789 1024; do
    python run_seed.py --seed "$seed" --method "$method"
  done
done
```

Each run writes `results/results_{method}_seed{seed}.json`.

On a cluster this can be dispatched as 10 independent SLURM jobs (one per
seed × method).

### 5. Aggregate and test for significance

After all 10 per-seed results exist:

```bash
python aggregate.py
```

This prints the per-class mean ± std table, the Wilcoxon signed-rank test
(one-sided, JMDS > unweighted on macro accuracy) and saves
`results/aggregated_results.json`.

### 6. Figures for the thesis

```bash
# (a) trains two models on the full data and dumps predictions / learning curves
python collect_viz.py          # or sbatch collect_viz.slurm on a cluster

# (b) turns viz_data.json + the committed intermediate files into PNGs
python generate_viz.py
```

The resulting PNGs land in `figures/` (pre-built copies are included so you
can see the expected output without running the pipeline).

## Hyper-parameters

| Component | Value |
|---|---|
| FastText | skip-gram, 150-dim, char n-grams `n=2..10`, Ancient Greek corpus |
| GMM | K=5, full covariance, `random_state=42` |
| Temperature | T=20 on GMM log-responsibilities |
| LPG | MINGAP normalised by corpus max |
| JMDS | clamped to `[0.05, 1.0]` |
| Student | XLM-RoBERTa-base |
| Optimiser | AdamW, lr=2e-5 |
| Batch size | 8 (train), 16 (eval) |
| Epochs | 3 |
| Seeds | 42, 123, 456, 789, 1024 |

## Notes

- The `LANGUAGE` label is retained in `LABELS` for backward compatibility with
  older data dumps; it is effectively empty in the cleaned NEReus split and
  does not appear in the reported test metrics.
- `run_seed.py` saves under `results/`; `aggregate.py` reads from there.
- If you re-run step (2), expect non-trivial cost and latency from the OpenAI
  Batch API (batches can take several hours to complete).
