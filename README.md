# Authorship Attribution: BERT and ModernBERT Feature Attribution Analysis

The following project is investigating what textual features drive correct classification
by fine-tuned BERT and ModernBERT authorship classifiers, with the goal of informing
future obfuscation strategies.

## Research questions

1. What token-level features (content words, punctuation, stop words) does the classifier rely on?
2. Are LLM-generated texts distinguishable from human-written texts, and what features differentiate them?
3. Can a reader identify the author/source class just by reading the text?
4. Which features could be targeted for future obfuscation strategies?
5. Does ModernBERT rely on different features than BERT, and why?


## Datasets

### PAN25 (LLM Detection)
- **Task:** 23-class classification (22 LLM sources + human)
- **Train:** 23,707 examples | **Val:** 3,556 examples
- **Genres:** fiction, essays, news
- **BERT balanced accuracy (best seed):** 62.3%
- **ModernBERT balanced accuracy (best seed):** 80.2%
- **Location:** `data/pan25/`

### AuthorMix (Author Attribution)
- **Task:** 14-class classification (literary authors, politicians, bloggers, AMT workers)
- **Train:** 14,579 examples | **Val:** 3,642 examples
- **Classes:** Fitzgerald, Hemingway, Woolf, Obama, Bush, Trump, 5 blog authors, 3 AMT workers
- **BERT balanced accuracy (best seed):** 86.2%
- **ModernBERT balanced accuracy (best seed):** 88.2%
- **Location:** `data/AuthorMix/`

## Models

### BERT
- Model: `bert-base-uncased`
- Max sequence length: 512 tokens (hard architectural limit)
  
### ModernBERT
- Model: `answerdotai/ModernBERT-base` (149M parameters)
- Max sequence length: 8,192 tokens (full document)
- Attribution cap: 1,500 tokens (covers >95% of PAN25 examples)


## Project structure

```
.
├── README.md
├── requirements.txt
│
├── notebooks/
│   ├── pan25_eda.ipynb                          # EDA - PAN25 dataset
│   ├── authormix_eda.ipynb                      # EDA - AuthorMix dataset
│   │
│   ├── baseline.ipynb                           # BERT fine-tuning and evaluation
│   ├── attribution.ipynb                        # BERT attribution (IG, SHAP, AR)
│   ├── attribution_enhanced.ipynb               # BERT attribution analysis
│   │
│   ├── baseline_modernbert.ipynb                # ModernBERT fine-tuning and evaluation
│   ├── attribution_modernbert.ipynb             # ModernBERT attribution (IG, SHAP, AR)
│   └── attribution_enhanced_modernbert.ipynb    # ModernBERT attribution analysis
│
├── results/
│   ├── balanced_accuracy_summary.csv            # BERT accuracy across seeds
│   ├── modernbert_vs_bert_summary.csv           # BERT vs ModernBERT comparison
│   │
│   ├── pan25_full_validation_*.csv / .json      # BERT attribution results - PAN25
│   ├── authormix_full_validation_*.csv / .json  # BERT attribution results - AuthorMix
│   │
│   ├── pan25_modernbert_*.csv / .json           # ModernBERT attribution results - PAN25
│   └── authormix_modernbert_*.csv / .json       # ModernBERT attribution results - AuthorMix
│
└── reports/
    ├── ATTRIBUTION_REPORT_BERT.pdf              # BERT attribution findings (PDF)
    └── ATTRIBUTION_REPORT_MODERNBERT.pdf        # ModernBERT attribution findings (PDF)
```


## Notebooks

### `pan25_eda.ipynb` and `authormix_eda.ipynb`
Exploratory data analysis before training. Covers class distribution, text length
analysis, truncation risk, duplicate detection, and dataset quality checks.
Run independently, no prior steps required.

### `baseline.ipynb` - BERT Fine-tuning
Multi-seed training of `bert-base-uncased` on both datasets.
- Tokenization with max length 512
- 3 random seeds, best seed selected by macro F1
- Saves per-seed metrics, best model checkpoints, and label maps
- Computes confusion matrices and per-class accuracy

### `baseline_modernbert.ipynb` - ModernBERT Fine-tuning
Same pipeline as `baseline.ipynb` but for `answerdotai/ModernBERT-base`.
- Max sequence length 1,500 for training (covers >95% of PAN25)
- Eager attention mode required for compatibility
- Mixed precision (fp16) for inference; fp32 for attribution

### `attribution.ipynb` - BERT Attribution
Runs three attribution methods on the full validation sets using the best BERT model.
- **Integrated Gradients (IG):** gradient-based, 15 interpolation steps
- **GradientSHAP:** gradient-based, 10 random baseline samples
- **Attention Rollout (AR):** manual Q/K computation (SDPA workaround)
- Saves ratios CSV and full attributions JSON per dataset

### `attribution_modernbert.ipynb` - ModernBERT Attribution
Same as `attribution.ipynb` but for ModernBERT. Key technical differences:
- Sequences truncated at 1,500 tokens (memory constraint: quadratic attention)
- `internal_batch_size=1` for IG (OOM prevention)
- Manual loop for GradientSHAP (no `internal_batch_size` parameter)
- `attention_mask.expand()` required (ModernBERT does not broadcast mask shapes)

### `attribution_enhanced.ipynb` - BERT Attribution analysis
Loads outputs from `attribution.ipynb` and performs 12 dimensions of analysis.
Covers token heatmaps, POS analysis, lexical sophistication, attribution
concentration, cross-method agreement, vocabulary fingerprints, positional
analysis, and statistical testing.

### `attribution_enhanced_modernbert.ipynb` - ModernBERT Attribution analysis
Same analysis pipeline as `attribution_enhanced.ipynb` adapted for ModernBERT.
Includes an additional section comparing ModernBERT content ratios to BERT.
Note: ModernBERT uses GPT-2 style tokenisation (Ġ prefix for word-initial tokens
instead of BERT's ## suffix for subword continuations).


## How to run

Upload the `data/` folder to `My Drive/ap-thesis/data/` (or adjust the `ROOT`
path variable at the top of each notebook).

Run in the following order:

```
# Exploratory (optional, run independently)
pan25_eda.ipynb
authormix_eda.ipynb

# BERT pipeline
1. baseline.ipynb                          → trains BERT, saves to runs/models/
2. attribution.ipynb                       → requires step 1
3. attribution_enhanced.ipynb              → requires step 2

# ModernBERT pipeline
4. baseline_modernbert.ipynb               → trains ModernBERT, saves to runs/models/
5. attribution_modernbert.ipynb            → requires step 4
6. attribution_enhanced_modernbert.ipynb   → requires step 5
```

Dependencies are installed inline in each notebook. For local runs:

```bash
pip install -r requirements.txt
```


## Key findings

### Performance

| Dataset | BERT (mean) | ModernBERT (mean) | Difference |
|---------|:-----------:|:-----------------:|:----------:|
| PAN25 (23 classes) | 0.615 | 0.797 | +17.8 pp |
| AuthorMix (14 classes) | 0.857 | 0.878 | +2.1 pp |

### What features drive classification?

**BERT - PAN25 (LLM detection):**
- Relies primarily on stop words and punctuation (content ratio 0.18-0.38)
- Human text has the lowest content ratio (0.23) - identified through function word patterns
- Attention Rollout is 95-99% punctuation - not semantically meaningful

**ModernBERT - PAN25 (LLM detection):**
- Relies on content words for every class (content ratio 0.50-0.78)
- Human text sits mid-range at 0.72 - identified through content vocabulary
- Attention Rollout gives 57-67% content words - semantically meaningful
- Strongest human vs. GPT-4o signal: AR stop ratio (Cohen's d = -0.59, p = 5.8e-12)

**Both models - AuthorMix (author attribution):**
- Rely on topic-specific content vocabulary (nouns dominant)
- Each author has a clear vocabulary fingerprint (Obama: policy terms, Hemingway: sensory nouns)
- Attribution is distributed across the full text, not just the opening

### Obfuscation implications

| Task | BERT target | ModernBERT target |
|------|------------|------------------|
| LLM detection | Add content words; rewrite opening | Replace specific content vocabulary; rewrite throughout |
| Author attribution | Replace topic nouns | Replace topic nouns + adjust stop word patterns across full text |


## Environment

- Python 3.10+
- Google Colab T4 GPU (16 GB VRAM)
- BERT: `bert-base-uncased`
- ModernBERT: `answerdotai/ModernBERT-base`

