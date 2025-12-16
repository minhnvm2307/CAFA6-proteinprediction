# CAFA-6 Protein function Prediction

> Team: Neo_Junai_INT34057

> Ranking: ~500

## Setup
### Prerequisites

- Python 3.13+

- CUDA-capable GPU (recommended)

- uv/miniconda

### Installation

1. Clone this repository:

2. Set up environment:

```
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate proteinprediction

# Install additional dependencies
uv sync
or
pip install -r requirements.txt
```

3. Download dataset

```
# Download zipped dataset from github-release
wget https://github.com/minhnvm2307/CAFA6-proteinprediction/releases/download/dataset/dataset.zip

# Unzip files directly into ./data
unzip dataset.zip -d data
```

## Key Dependencies

- Deep Learning: PyTorch, TensorFlow, Transformers
- Protein Analysis: BioPython, Fair-ESM, ProteinBERT
- GO Analysis: GOAtools
- ML Tools: Scikit-learn


## Approach

### 1. Baseline Methods

- BLAST-based similarity search
- Feature engineering (amino acid composition, physicochemical properties)
- Traditional ML models (SVM, KNN)

### 2. Advanced Methods

- Protein Language Models (ESM, ProtBERT, ProtT5)
- MLP architect
- Ensemble 

## Running

Run everything from the repo root with your environment activated (`conda activate proteinprediction` or `source .venv/bin/activate`).

### 1) Data prep / EDA

- Notebook: open `preprocessing/cafa-6-analysis.ipynb` to explore distributions and filtering choices.
- Quick preprocessing sanity check (loads data, applies default filters, demonstrates embedding loading):
```
python -m preprocessing.data_preprocessing
```
  This expects training data under `data/Train/` and will print shapes plus a small sample.

### 2) BLAST baseline

- Install BLAST+ binaries and ProFun dependencies per `blast/README.md`.
- Generate BLAST predictions for `data/Test/testsuperset.fasta`:
```
python -m blast.run_blastp
```
  Creates `blast_pred.tsv` in the repo root.

### 3) Model training/prediction (flat scripts)

All entrypoints live under `models/` and accept `--mode train`, `--mode predict`, or `--mode train_and_predict` (default). Logging is enabled by default.

Traditional baselines (CTD + dipeptide features from FASTA):
```
# KNN baseline
python models/baseline_knn.py --data-root data --submission submission_knn.tsv

# Linear SVM baseline
python models/baseline_svm.py --data-root data --submission submission_svm.tsv
```

Embedding MLPs (precomputed embeddings under `data/embedding/{esm,prottrans,protbert}`):
```
# ESM2 MLP
python models/esm2_mlp.py --data-root data --submission submission_esm.tsv

# ProtT5 MLP
python models/prott5_mlp.py --data-root data --submission submission_prott5.tsv

# ProtBERT MLP
python models/protbert_mlp.py --data-root data --submission submission_protbert.tsv
```
Common flags: `--epochs`, `--batch-size`, `--lr`, `--threshold`, `--model-path`.

Branch-based ensembling of submission files:
```
python models/ensemble_mlp.py \
  --go-path data/Train/go-basic.obo \
  --submissions data/checkpoint/submissions/submission_t5.tsv data/checkpoint/submissions/submission_blast.tsv \
  --output ensemble_submission.tsv
```

### 4) Model cheat sheet

| Model                    | Features / Notes            | Status / Metric |
|--------------------------|-----------------------------|-----------------|
| SVM + Blast              | Handcrafted features        | 0.140           |
| KNN                      | CTD + dipeptide             | 0.138           |
| KNN + BLAST              | Hybrid similarity           | 0.195           |
| MLP + ProtBert           | Full class, no filter       | 0.144           |
| MLP + ProtT5             | Outliers filtering          | 0.208           |
| MLP + ESM2               | Outliers filtering          | 0.192           |
| MLP + ESM2 + Regulation  | Embeddings + regularization | 0.195           |
| Ensemble (BLAST + ESM2)  | Branch bonus ensemble       | 0.243           |
| Ensemble (BLAST + ProtT5)| Branch bonus ensemble       | **Best score: 0.252** |
