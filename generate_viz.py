"""
Generate all 5 visualizations for thesis.
Run after viz_data.json has been downloaded from the cluster.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs('figures', exist_ok=True)

LABELS    = ["O", "PERSON", "GOD", "LOC", "NORP"]
LABEL_CLR = {"O": "#aec7e8", "PERSON": "#1f77b4", "GOD": "#ff7f0e",
             "LOC": "#2ca02c", "NORP": "#d62728", "LANGUAGE": "#9467bd"}

plt.rcParams.update({'font.size': 12, 'figure.dpi': 150,
                     'axes.spines.top': False, 'axes.spines.right': False})

# Load data
with open('data/combined_confidence.json', encoding='utf-8') as f:
    cc = json.load(f)          # pseudo-labels + JMDS scores (train split)
with open('data/train_data.json', encoding='utf-8') as f:
    td = json.load(f)          # true labels (train split, aligned with cc)
with open('data/fasttext_emb.json', encoding='utf-8') as f:
    ft_emb = json.load(f)      # word → 150-dim vector
with open('viz_data.json', encoding='utf-8') as f:
    viz = json.load(f)         # predictions + learning curves from cluster

print("Data loaded.")

# Flatten token-level data for train split
tokens_all, pseudo_all, true_all, conf_all = [], [], [], []
for sent_cc, sent_td in zip(cc, td):
    for tok, pseudo, true, conf in zip(sent_cc['tokens'], sent_cc['ner_tags'],
                                       sent_td['ner_tags'], sent_cc['confidences']):
        tokens_all.append(tok)
        pseudo_all.append(pseudo)
        true_all.append(true)
        conf_all.append(conf)

tokens_all = np.array(tokens_all)
pseudo_all = np.array(pseudo_all)
true_all   = np.array(true_all)
conf_all   = np.array(conf_all, dtype=float)

print(f"Total train tokens: {len(tokens_all)}")

# (a) Confusion matrix for JMDS + FastText, row-normalised
fig, ax = plt.subplots(figsize=(8, 6))

preds  = viz['predictions']['jmds']['preds']
labels = viz['predictions']['jmds']['labels']

present = [l for l in LABELS if l in set(labels) | set(preds)]
cm = confusion_matrix(labels, preds, labels=present)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=present, yticklabels=present,
            vmin=0, vmax=1, ax=ax, linewidths=0.5,
            cbar_kws={'label': 'Proportion'})
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
ax.set_title('JMDS + FastText — Normalised Confusion Matrix\non NEReus Test Set (seed=42)',
             fontsize=13, fontweight='bold')
ax.tick_params(axis='x', rotation=30)

plt.tight_layout()
fig.savefig('figures/a_confusion_matrix.png', bbox_inches='tight')
plt.close()
print("Saved: figures/a_confusion_matrix.png")

# (b) JMDS confidence distribution by true label (violin + strip)
fig, ax = plt.subplots(figsize=(10, 5))

plot_labels = [l for l in LABELS if l in true_all]
data_by_label = [conf_all[true_all == l] for l in plot_labels]
counts        = [np.sum(true_all == l) for l in plot_labels]

parts = ax.violinplot(data_by_label, positions=range(len(plot_labels)),
                      showmedians=True, showextrema=True)

for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(LABEL_CLR.get(plot_labels[i], '#888'))
    pc.set_alpha(0.7)
parts['cmedians'].set_color('black')
parts['cmedians'].set_linewidth(2)

# overlay means as dots
for i, d in enumerate(data_by_label):
    ax.scatter(i, np.mean(d), color='red', zorder=5, s=60, label='Mean' if i == 0 else '')

ax.set_xticks(range(len(plot_labels)))
ax.set_xticklabels([f'{l}\n(n={c:,})' for l, c in zip(plot_labels, counts)], fontsize=11)
ax.set_ylabel('JMDS Confidence Score', fontsize=12)
ax.set_xlabel('True NER Label', fontsize=12)
ax.set_title('JMDS Confidence Score Distribution by True Label (Train Split)', fontsize=13, fontweight='bold')
ax.axhline(np.mean(conf_all), color='gray', linestyle='--', alpha=0.6, label=f'Global mean ({np.mean(conf_all):.3f})')
ax.legend()
ax.set_ylim(0, 1.05)

plt.tight_layout()
fig.savefig('figures/b_jmds_distribution.png', bbox_inches='tight')
plt.close()
print("Saved: figures/b_jmds_distribution.png")

# (c) Pseudo-label quality: per-class accuracy and stacked assignment bars
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: pseudo-label accuracy per true class
accs, ns = [], []
for lbl in LABELS:
    mask = true_all == lbl
    n    = mask.sum()
    if n == 0:
        accs.append(0); ns.append(0); continue
    acc = (pseudo_all[mask] == true_all[mask]).mean()
    accs.append(acc); ns.append(int(n))

bars = axes[0].bar(LABELS, accs,
                   color=[LABEL_CLR.get(l, '#888') for l in LABELS],
                   edgecolor='white', linewidth=1.5)
for bar, acc, n in zip(bars, accs, ns):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{acc:.2f}\n(n={n:,})', ha='center', va='bottom', fontsize=9)
axes[0].set_ylim(0, 1.15)
axes[0].set_ylabel('Pseudo-Label Accuracy', fontsize=12)
axes[0].set_xlabel('True Label', fontsize=12)
axes[0].set_title('GPT Pseudo-Label Accuracy\nper True Entity Class', fontsize=12, fontweight='bold')
axes[0].axhline(1.0, color='gray', linestyle='--', alpha=0.4)

# Right: stacked bar — for each true class, what pseudo-label did GPT assign?
plot_lbls = [l for l in LABELS if (true_all == l).sum() > 0]
bottom = np.zeros(len(plot_lbls))
all_assigned = sorted(set(pseudo_all))
cmap   = plt.get_cmap('tab10')

for j, assigned in enumerate(all_assigned):
    fracs = []
    for lbl in plot_lbls:
        mask  = true_all == lbl
        total = mask.sum()
        frac  = (pseudo_all[mask] == assigned).sum() / total if total > 0 else 0
        fracs.append(frac)
    axes[1].bar(plot_lbls, fracs, bottom=bottom,
                label=assigned, color=cmap(j / len(all_assigned)), alpha=0.85)
    bottom += np.array(fracs)

axes[1].set_ylim(0, 1.05)
axes[1].set_ylabel('Proportion of tokens', fontsize=12)
axes[1].set_xlabel('True Label', fontsize=12)
axes[1].set_title('What GPT Assigned\n(stacked by assigned pseudo-label)', fontsize=12, fontweight='bold')
axes[1].legend(title='Assigned', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)

fig.suptitle('Pseudo-Label Quality Analysis (Train Split)', fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig('figures/c_pseudolabel_quality.png', bbox_inches='tight')
plt.close()
print("Saved: figures/c_pseudolabel_quality.png")

# (d) t-SNE projection coloured by true label and JMDS confidence (side by side).
# Only tokens that exist in the FastText vocabulary, stratified sample of 4,000.
np.random.seed(42)
vocab_mask = np.array([t in ft_emb for t in tokens_all])
idx_avail  = np.where(vocab_mask)[0]

# stratified sample: up to 4000 total, preserving entity/O ratio
entity_idx = idx_avail[true_all[idx_avail] != 'O']
o_idx      = idx_avail[true_all[idx_avail] == 'O']
n_total    = 4000
if len(entity_idx) >= n_total:
    sampled_e  = np.random.choice(entity_idx, n_total, replace=False)
    sample_idx = sampled_e
else:
    n_o_sample = min(len(o_idx), n_total - len(entity_idx))
    sampled_o  = np.random.choice(o_idx, n_o_sample, replace=False)
    sample_idx = np.concatenate([entity_idx, sampled_o])
np.random.shuffle(sample_idx)

X   = np.array([ft_emb[tokens_all[i]] for i in sample_idx])
y   = true_all[sample_idx]
c_s = conf_all[sample_idx]

print(f"t-SNE on {len(X)} tokens…")
tsne  = TSNE(n_components=2, random_state=42, perplexity=40, max_iter=1000)
X_2d  = tsne.fit_transform(X)
print("t-SNE done.")

# Panel 1: coloured by true label
fig1, ax1 = plt.subplots(figsize=(8, 6))
present_lbls = [l for l in LABELS if l in y]
for lbl in present_lbls:
    mask = y == lbl
    ax1.scatter(X_2d[mask, 0], X_2d[mask, 1],
                c=LABEL_CLR.get(lbl, '#888'), label=lbl,
                s=8, alpha=0.6, linewidths=0)
ax1.set_title('FastText Embedding Space (t-SNE)\nColoured by True NER Label',
              fontsize=12, fontweight='bold')
ax1.legend(markerscale=3, fontsize=10)
ax1.set_xlabel('t-SNE 1'); ax1.set_ylabel('t-SNE 2')
plt.tight_layout()
fig1.savefig('figures/d_tsne_labels.png', bbox_inches='tight')
plt.close()
print("Saved: figures/d_tsne_labels.png")

# Panel 2: coloured by JMDS confidence score
fig2, ax2 = plt.subplots(figsize=(8, 6))
sc = ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=c_s,
                 cmap='RdYlGn', s=8, alpha=0.6,
                 vmin=0, vmax=1, linewidths=0)
plt.colorbar(sc, ax=ax2, label='JMDS Confidence Score')
ax2.set_title('FastText Embedding Space (t-SNE)\nColoured by JMDS Confidence Score',
              fontsize=12, fontweight='bold')
ax2.set_xlabel('t-SNE 1'); ax2.set_ylabel('t-SNE 2')
plt.tight_layout()
fig2.savefig('figures/d_tsne_confidence.png', bbox_inches='tight')
plt.close()
print("Saved: figures/d_tsne_confidence.png")

# (e) Learning curves: eval loss per epoch, across seeds, for both methods
lc = viz['learning_curves']
epochs = [1, 2, 3]
fig, ax = plt.subplots(figsize=(9, 5))

for method, color, label in [('unweighted', '#1f77b4', 'Unweighted'),
                               ('jmds',       '#ff7f0e', 'JMDS+FastText')]:
    all_losses = []
    for key, val in lc.items():
        if method in key:
            # sort by epoch, extract loss values for epochs 1,2,3
            losses_by_epoch = {round(v['epoch']): v['eval_loss'] for v in val}
            run_losses = [losses_by_epoch.get(e, np.nan) for e in epochs]
            all_losses.append(run_losses)
            ax.plot(epochs, run_losses, color=color, alpha=0.25, linewidth=1)

    if all_losses:
        arr  = np.array(all_losses)
        mean = np.nanmean(arr, axis=0)
        std  = np.nanstd(arr, axis=0)
        ax.plot(epochs, mean, color=color, linewidth=2.5, label=f'{label} (mean)')
        ax.fill_between(epochs, mean - std, mean + std,
                        color=color, alpha=0.15, label=f'{label} ±1 std')

ax.set_xticks(epochs)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Eval Loss', fontsize=12)
ax.set_title('Eval Loss per Epoch — 5 Seeds × 2 Methods', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
plt.tight_layout()
fig.savefig('figures/e_learning_curves.png', bbox_inches='tight')
plt.close()
print("Saved: figures/e_learning_curves.png")

print("\nAll figures saved to ./figures/")
