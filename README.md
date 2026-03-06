# Authorship Attribution with BERT: Feature Attribution Analysis

AP Thesis project investigating what textual features drive correct classification by a fine-tuned BERT authorship classifier, with the goal of informing future obfuscation strategies.

## Research Questions

1. What token-level features (content words, punctuation, stop words) does the BERT classifier rely on for authorship attribution?
2. Are LLM-generated texts distinguishable from human-written texts, and what features differentiate them?
3. Can a reader identify the author/source class just by reading the text?
4. Which features could be targeted for future obfuscation strategies?

## Datasets

### PAN25 (LLM Detection)
- **Task:** 23-class classification (22 LLM sources + human)
- **Train:** 23,707 examples | **Val:** 3,589 examples
- **Genres:** fiction, essays, news
- **Best-seed balanced accuracy:** 68.0%
- **Location:** `data/pan25/`

### AuthorMix (Author Attribution)
- **Task:** 14-class classification (literary authors, politicians, bloggers, AMT workers)
- **Train:** 14,579 examples | **Val:** 3,642 examples
- **Categories:** author (Fitzgerald, Hemingway, Woolf), blog, speech (Obama, Bush, Trump), AMT (h, pp, qq)
- **Best-seed balanced accuracy:** 92.6%
- **Location:** `data/style-remix/AuthorMix/`

## Project Structure

```
.
├── README.md
├── requirements.txt
│
├── baseline.ipynb                  # BERT fine-tuning & multi-seed training
├── attribution.ipynb               # Token-level attribution (IG, SHAP, AR) on full validation
├── attribution_enhanced.ipynb      # Visualization & statistical analysis of attributions
├── pan25_eda.ipynb                 # Exploratory data analysis — PAN25 dataset
├── authormix_eda.ipynb             # Exploratory data analysis — AuthorMix dataset
│
├── data/
│   ├── pan25/
│   │   ├── train.jsonl             # PAN25 training set
│   │   └── val.jsonl               # PAN25 validation set
│   └── style-remix/
│       └── AuthorMix/
│           ├── AuthorMix-train.json
│           ├── AuthorMix-val.json
│           └── AuthorMix-test.json
│
├── results/                         
│   ├── balanced_accuracy_summary.csv
│   ├── *_per_class_accuracy_validation.csv
│   ├── *_full_validation_all_methods_ratios.csv
│   ├── *_full_validation_all_methods_attributions.json
│   ├── *_ratio_summary.csv
│   └── *_agreement_summary.csv
│
│
└── ATTRIBUTION_REPORT.md           # key findings
```

## Notebooks

### 1. `baseline.ipynb` — BERT Fine-Tuning

Multi-seed training of `bert-base-uncased` for sequence classification on both datasets.

- Tokenization with max length 512
- Multi-seed training (selects best seed by F1 macro)
- Saves per-seed metrics, best model checkpoints, and label maps
- Computes confusion matrices and classification reports

### 2. `pan25_eda.ipynb` — PAN25 Exploratory Data Analysis

Data quality assessment and risk identification before training.

- Class distribution (35x imbalance: human vs smallest LLM class)
- Genre-model confound analysis (many models appear in only 1 genre)
- Text length distributions and BERT truncation risk (85% of texts truncated)
- Duplicate detection and train-val overlap check
- Human vs LLM statistical comparison (word length, TTR, sentence length)
- Risk summary dashboard with severity ratings

### 3. `authormix_eda.ipynb` — AuthorMix Exploratory Data Analysis

- Class distribution (68x imbalance: fitzgerald 885 vs qq 13 in val)
- Category-style confound (every style maps to one genre — CRITICAL risk)
- Very short text analysis (3% are < 10 words, mostly speech fragments)
- Within-category class separability analysis
- AMT class deep dive (13-17 val samples per class)
- Risk summary dashboard with severity ratings

### 4. `attribution.ipynb` — Feature Attribution Computation

Runs three attribution methods on the full validation sets of both datasets.

- **Integrated Gradients (IG)** — gradient-based (Captum), 15 steps
- **GradientSHAP** — gradient-based (Captum), 10 samples
- **Attention Rollout (AR)** — manual Q/K computation (SDPA workaround, no `output_attentions`)
- Balanced accuracy and per-class accuracy computation
- Saves ratios CSV + full attributions JSON per dataset

### 5. `attribution_enhanced.ipynb` — Attribution Analysis & Visualization

Loads outputs from `attribution.ipynb` and performs 11 dimensions of analysis:

1. Token-level heatmap visualizations
2. POS-tag distribution analysis
3. etc.


## How to Run

All notebooks are designed for **Google Colab** with a T4 GPU.

### Step 1 — Upload data to Google Drive

Upload the `data/` folder to `My Drive/ap-thesis/data/` (or adjust `DATA_DIR` / `ROOT` paths in each notebook).

### Step 2 — Run in order

```
1. baseline.ipynb          → trains BERT, saves models to runs/models/
2. pan25_eda.ipynb         → EDA (independent, can run anytime)
3. authormix_eda.ipynb     → EDA (independent, can run anytime)
4. attribution.ipynb       → requires trained models from step 1
5. attribution_enhanced.ipynb → requires outputs from step 4
```

### Step 3 — Install dependencies

Dependencies are installed inline via `!pip install` in each notebook. For local runs:

```bash
pip install -r requirements.txt
```

## Environment

- Python 3.10+
- Google Colab (T4 GPU recommended)
- BERT model: `bert-base-uncased`
