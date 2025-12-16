"""
Classic ML baselines packaged as classes.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.decomposition import PCA
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.linear_model import SGDClassifier

logger = logging.getLogger(__name__)


# --- Feature extraction helpers (CTD + dipeptide) ---
AMINO_ACIDS = "ARNDCQEGHILKMFPSTWYV"
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

CTD_GROUPS_BY_LIST = {
    "Hydrophobicity": {
        1: {"A", "V", "L", "I", "M", "F", "W", "C"},
        2: {"G", "H", "Y", "P", "T", "S"},
        3: {"R", "K", "Q", "E", "D", "N"},
    },
    "Charge": {
        1: {"D", "E"},
        2: {"A", "G", "I", "L", "M", "F", "P", "Q", "S", "T", "W", "Y", "V", "N", "C"},
        3: {"K", "R", "H"},
    },
    "VanDerWaals": {
        1: {"A", "G", "S", "C"},
        2: {"T", "D", "P", "N", "V"},
        3: {"E", "Q", "L", "I", "F", "Y", "M", "H", "K", "R", "W"},
    },
    "Polarity": {
        1: {"L", "A", "W", "F", "C", "M", "V", "I", "Y"},
        2: {"P", "T", "S", "G", "H"},
        3: {"Q", "N", "E", "D", "K", "R"},
    },
    "Polarizability": {
        1: {"G", "A", "S", "D", "C"},
        2: {"T", "P", "N", "H", "E", "Q", "K"},
        3: {"M", "I", "L", "V", "F", "Y", "W", "R"},
    },
    "SecondStructure": {
        1: {"E", "A", "L", "M", "Q", "K", "R", "H"},
        2: {"V", "I", "Y", "C", "W", "F", "T"},
        3: {"G", "N", "P", "S", "D"},
    },
    "Solvent": {
        1: {"A", "L", "F", "C", "G", "I", "V", "W"},
        2: {"R", "K", "Q", "E", "D", "N"},
        3: {"M", "S", "P", "T", "H", "Y"},
    },
}


def naive_freq(sequence: str) -> List[float]:
    freq = [0] * 20
    for aa in sequence:
        idx = AA_TO_INDEX.get(aa)
        if idx is not None:
            freq[idx] += 1
    total = len(sequence)
    if total:
        return [c / total for c in freq]
    return freq


def get_group(aa: str, property_map) -> int | None:
    for g, aa_set in property_map.items():
        if aa in aa_set:
            return g
    return None


def aa_ctd(sequence: str, physicochem: str) -> List[float]:
    property_map = CTD_GROUPS_BY_LIST[physicochem]
    groups = [get_group(aa, property_map) for aa in sequence]
    groups = [g for g in groups if g]
    L = len(groups)
    if L == 0:
        return [0.0] * 21

    counts = {1: 0, 2: 0, 3: 0}
    for g in groups:
        counts[g] += 1
    composition = [counts[1] / L, counts[2] / L, counts[3] / L]

    T12 = T13 = T23 = 0
    for g1, g2 in zip(groups[:-1], groups[1:]):
        if g1 == g2:
            continue
        lo, hi = sorted([g1, g2])
        if (lo, hi) == (1, 2):
            T12 += 1
        elif (lo, hi) == (1, 3):
            T13 += 1
        elif (lo, hi) == (2, 3):
            T23 += 1
    denom = max(L - 1, 1)
    transition = [T12 / denom, T13 / denom, T23 / denom]

    positions = {1: [], 2: [], 3: []}
    for idx, g in enumerate(groups, start=1):
        positions[g].append(idx)
    distribution: List[float] = []
    for g in [1, 2, 3]:
        pos_list = positions[g]
        if not pos_list:
            distribution.extend([0.0] * 5)
            continue
        Ng = len(pos_list)
        for pk in [0.0, 0.25, 0.5, 0.75, 1.0]:
            if pk == 0:
                distribution.append(pos_list[0] / L)
            else:
                idx = max(int(np.ceil(Ng * pk)) - 1, 0)
                distribution.append(pos_list[idx] / L)
    return composition + transition + distribution


def dipeptide_composition(seq: str) -> List[float]:
    AA = "ACDEFGHIKLMNPQRSTVWY"
    dipeptides = [a + b for a in AA for b in AA]
    seq = seq.upper()
    length = len(seq) - 1 if len(seq) > 1 else 1
    return [sum(1 for i in range(len(seq) - 1) if seq[i] + seq[i + 1] == dp) / length for dp in dipeptides]


def extract_accession(header: str) -> str:
    parts = header.lstrip(">").split("|")
    if len(parts) >= 2:
        return parts[1]
    return header.lstrip(">")


BIO_PROPERTIES = ["Hydrophobicity", "Charge", "VanDerWaals", "Polarity", "Polarizability", "SecondStructure", "Solvent"]


def load_fasta_features(fasta_path: Path) -> Tuple[List[str], np.ndarray]:
    ids: List[str] = []
    feats: List[List[float]] = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        pid = extract_accession(record.id)
        seq = str(record.seq).upper()
        x: List[float] = []
        x.extend(naive_freq(seq))
        for prop in BIO_PROPERTIES:
            x.extend(aa_ctd(seq, prop))
        x.extend(dipeptide_composition(seq))
        ids.append(pid)
        feats.append(x)
    return ids, np.vstack(feats)


def load_label_dict(train_terms_path: Path) -> Dict[str, List[str]]:
    df = pd.read_csv(train_terms_path, sep="\t", names=["EntryID", "GO", "Ont"])
    grouped = df.groupby("EntryID")["GO"].apply(list)
    return grouped.to_dict()


def build_label_matrix(train_ids: Sequence[str], label_dict: Dict[str, List[str]]):
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform([label_dict.get(pid, []) for pid in train_ids])
    return y, mlb


@dataclass
class TraditionalConfig:
    data_root: Path = Path("data")
    pca_components: int = 100
    n_neighbors: int = 5
    threshold: float = 0.01
    batch_size: int = 512
    model_path: Path = Path("data/checkpoint/traditional_model.joblib")
    submission_path: Path = Path("submission.tsv")


class KNNModel:
    def __init__(self, config: TraditionalConfig = TraditionalConfig()):
        self.config = config
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.config.pca_components)
        self.model = KNeighborsClassifier(n_neighbors=config.n_neighbors, metric="euclidean", weights="distance", n_jobs=-1)
        self.mlb: MultiLabelBinarizer | None = None
        self.train_ids: List[str] = []
        self.test_ids: List[str] = []

    def train(self) -> None:
        data_root = self.config.data_root
        logger.info("Loading training FASTA and labels for KNN")
        train_ids, X_train = load_fasta_features(data_root / "Train" / "train_sequences.fasta")
        label_dict = load_label_dict(data_root / "Train" / "train_terms.tsv")
        y_train, self.mlb = build_label_matrix(train_ids, label_dict)
        self.train_ids = train_ids

        logger.info("Fitting scaler and PCA (components=%s)", self.config.pca_components)
        X_scaled = self.scaler.fit_transform(X_train)
        X_reduced = self.pca.fit_transform(X_scaled)
        logger.info("Training KNN (k=%d)", self.config.n_neighbors)
        self.model.fit(X_reduced, y_train)
        joblib.dump({"scaler": self.scaler, "pca": self.pca, "model": self.model, "classes": self.mlb.classes_}, self.config.model_path)
        logger.info("Saved KNN checkpoint to %s", self.config.model_path)

    def predict(self, output_path: Path | None = None) -> Path:
        if output_path is None:
            output_path = self.config.submission_path
        output_path = Path(output_path)

        logger.info("Loading KNN checkpoint from %s", self.config.model_path)
        checkpoint = joblib.load(self.config.model_path)
        self.scaler = checkpoint["scaler"]
        self.pca = checkpoint["pca"]
        self.model = checkpoint["model"]
        classes = checkpoint["classes"]
        self.mlb = MultiLabelBinarizer(classes=classes)
        self.mlb.fit([classes])

        logger.info("Loading test FASTA for KNN prediction")
        test_ids, X_test = load_fasta_features(self.config.data_root / "Test" / "testsuperset.fasta")
        self.test_ids = test_ids
        X_test_reduced = self.pca.transform(self.scaler.transform(X_test))
        prob_list = self.model.predict_proba(X_test_reduced)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Writing KNN predictions to %s", output_path)
        with open(output_path, "w") as f:
            for term_idx, term in enumerate(self.mlb.classes_):
                term_probs = prob_list[term_idx][:, 1] if prob_list[term_idx].shape[1] > 1 else prob_list[term_idx][:, 0]
                for pid, score in zip(test_ids, term_probs):
                    if score >= self.config.threshold:
                        f.write(f"{pid}\t{term}\t{float(score):.4f}\n")
        return output_path


class SVMModel:
    def __init__(self, config: TraditionalConfig = TraditionalConfig()):
        self.config = config
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)
        base = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3, class_weight="balanced", random_state=42)
        self.model = MultiOutputClassifier(base, n_jobs=-1)
        self.mlb: MultiLabelBinarizer | None = None

    def train(self) -> None:
        data_root = self.config.data_root
        logger.info("Loading training FASTA and labels for SVM")
        train_ids, X_train = load_fasta_features(data_root / "Train" / "train_sequences.fasta")
        label_dict = load_label_dict(data_root / "Train" / "train_terms.tsv")
        y_train, self.mlb = build_label_matrix(train_ids, label_dict)

        logger.info("Fitting scaler and PCA for SVM")
        X_scaled = self.scaler.fit_transform(X_train)
        X_reduced = self.pca.fit_transform(X_scaled)
        logger.info("Training linear SVM (multi-output)")
        self.model.fit(X_reduced, y_train)
        joblib.dump({"scaler": self.scaler, "pca": self.pca, "model": self.model, "classes": self.mlb.classes_}, self.config.model_path)
        logger.info("Saved SVM checkpoint to %s", self.config.model_path)

    def predict(self, output_path: Path | None = None) -> Path:
        if output_path is None:
            output_path = self.config.submission_path
        output_path = Path(output_path)

        logger.info("Loading SVM checkpoint from %s", self.config.model_path)
        checkpoint = joblib.load(self.config.model_path)
        self.scaler = checkpoint["scaler"]
        self.pca = checkpoint["pca"]
        self.model = checkpoint["model"]
        classes = checkpoint["classes"]
        self.mlb = MultiLabelBinarizer(classes=classes)
        self.mlb.fit([classes])

        logger.info("Loading test FASTA for SVM prediction")
        test_ids, X_test = load_fasta_features(self.config.data_root / "Test" / "testsuperset.fasta")
        X_test_reduced = self.pca.transform(self.scaler.transform(X_test))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Writing SVM predictions to %s", output_path)
        with open(output_path, "w") as f:
            prob_list = self.model.predict_proba(X_test_reduced)
            for term_idx, term in enumerate(self.mlb.classes_):
                term_probs = prob_list[term_idx][:, 1] if prob_list[term_idx].shape[1] > 1 else prob_list[term_idx][:, 0]
                for pid, score in zip(test_ids, term_probs):
                    if score >= self.config.threshold:
                        f.write(f"{pid}\t{term}\t{float(score):.4f}\n")
        return output_path
