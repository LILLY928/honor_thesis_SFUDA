# JMDS + FastText for Source-Free Domain Adaptation on Ancient Greek NER

Code and experiments for a joint model-data confidence (JMDS) framework that
weights pseudo-labels produced by a zero-shot LLM (GPT-4o-mini) with a
structural confidence signal derived from GMM clustering of FastText
embeddings, then fine-tunes XLM-RoBERTa-base on the NEReus Ancient Greek
corpus.


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

## Setup

Python 3.10+ recommended.

```bash
python -m venv venv
source venv/bin/activate       # on Windows: venv\Scripts\activate
pip install -r requirements.txt
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

Run all cells of `4_model_finetuning.ipynb`, or
from the command line:

```bash
python run_finetuning.py
```

### 5. Aggregate and test for significance

After all 10 per-seed results exist:

```bash
python aggregate.py
```

This prints the per-class mean ± std table, the Wilcoxon signed-rank test
(one-sided, JMDS > unweighted on macro accuracy) and saves
`results/aggregated_results.json`.
