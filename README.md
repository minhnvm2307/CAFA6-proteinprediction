# CAFA-6 Protein function Prediction

> Team: Neo_Junai_INT34057
> Ranking: 510

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

### 3) Ensembling example

- Place submission files in `data/checkpoint/submissions/` and adjust the list in `ensemble/run_example.py` if needed.
- Combine them with the branch-based ensemble:
```
python ensemble/run_example.py
```
  Writes `ensemble_submission.tsv`.

### 4) Traditional ML models

| Model        | Features / Notes     | Fmax  |
|--------------|----------------------|-------|
| SVM + Blast  | Handcrafted features | 0.140  |
| KNN          | k = 200, Sequence embedding              | 0.138  |
| KNN + BLAST  | Hybrid similarity    | 0.195   |

### 5) Advanced models

| Model                    | Features / Notes            | Status / Metric |
|--------------------------|-----------------------------|-----------------|
| MLP + ProtBert  | Full class, no filter | 0.144    |
| MLP + ProtT5  | Outliers filtering   | 0.208             |
| MLP + ESM2    | Outliers filtering  | 0.192             |
| MLP + ESM2 + Regulation  | Embeddings + regularization | 0.195             |
| Ensemble (BLAST + ESM2)| Branch bonus ensemble       | 0.243 |
| Ensemble (BLAST + ProtT5)| Branch bonus ensemble       | **Best score: 0.252** |
