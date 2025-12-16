"""
Embedding-based MLP trainers for CAFA-6.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, Dataset, TensorDataset

from preprocessing.data_preprocessing import ProteinDataProcessor
from preprocessing.utils import create_dataset_with_emb


logger = logging.getLogger(__name__)


class ProteinClassifier(nn.Module):
    """Simple MLP head for multi-label classification."""

    def __init__(self, input_dim: int, num_classes: int, dropout_rate: float = 0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        self.X = torch.tensor(embeddings, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


@dataclass
class EmbeddingModelConfig:
    emb_type: str  # {"esm", "prottrans", "protbert"}
    data_root: Path = Path("data")
    batch_size: int = 128
    num_epochs: int = 20
    learning_rate: float = 1e-3
    dropout: float = 0.3
    threshold: float = 0.01
    model_path: Path = Path("data/checkpoint/cafa_model.pth")
    submission_path: Path = Path("submission.tsv")


class EmbeddingMLPModel:
    """
    Train and predict a multi-label MLP on precomputed protein embeddings.
    """

    def __init__(self, config: EmbeddingModelConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mlb: Optional[MultiLabelBinarizer] = None
        self.model: Optional[ProteinClassifier] = None
        self.train_ids: list[str] = []
        self.test_ids: list[str] = []
        self.train_emb: Optional[np.ndarray] = None
        self.test_emb: Optional[np.ndarray] = None

    def _load_train_df(self):
        processor = ProteinDataProcessor(str(self.config.data_root) + "/", "str")
        train_df, _ = processor.prepare_training_data()
        return train_df

    def _load_embeddings(self, train_df):
        test_ids_path = self.config.data_root / "embedding" / self.config.emb_type / "test_ids.npy"
        self.test_ids = np.load(test_ids_path, allow_pickle=True)
        self.train_ids = train_df["ID"].to_list()
        self.train_emb, self.test_emb = create_dataset_with_emb(self.train_ids, self.test_ids, self.config.emb_type)

    def _build_labels(self, train_df):
        self.mlb = MultiLabelBinarizer()
        labels = self.mlb.fit_transform(train_df["term"])
        return labels

    def _build_model(self, input_dim: int, num_classes: int) -> ProteinClassifier:
        return ProteinClassifier(input_dim=input_dim, num_classes=num_classes, dropout_rate=self.config.dropout).to(
            self.device
        )

    def train(self) -> None:
        logger.info("Loading training data and embeddings for %s", self.config.emb_type)
        train_df = self._load_train_df()
        self._load_embeddings(train_df)
        y_train = self._build_labels(train_df)
        X_train = self.train_emb

        self.model = self._build_model(input_dim=X_train.shape[1], num_classes=y_train.shape[1])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.BCEWithLogitsLoss()

        train_loader = DataLoader(
            EmbeddingDataset(X_train, y_train), batch_size=self.config.batch_size, shuffle=True
        )

        self.model.train()
        logger.info("Starting training for %d epochs (batch=%d, lr=%s)", self.config.num_epochs, self.config.batch_size, self.config.learning_rate)
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            logger.info("Epoch %d/%d - loss: %.4f", epoch + 1, self.config.num_epochs, epoch_loss / max(len(train_loader), 1))

        self.save(self.config.model_path)
        logger.info("Saved weights to %s", self.config.model_path)

    def save(self, path: Path | str) -> None:
        if self.model is None:
            raise RuntimeError("Model not trained/loaded")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "classes": self.mlb.classes_ if self.mlb else None,
            },
            path,
        )

    def load(self, path: Path | str) -> None:
        path = Path(path)
        logger.info("Loading checkpoint from %s", path)
        checkpoint = torch.load(path, map_location=self.device)
        classes = checkpoint.get("classes")
        if classes is not None:
            self.mlb = MultiLabelBinarizer(classes=classes)
            self.mlb.fit([classes])

        if self.test_emb is None or len(self.test_emb.shape) != 2:
            raise RuntimeError("Load embeddings via predict() before calling load().")
        num_classes = len(classes) if classes is not None else self.test_emb.shape[1]
        self.model = self._build_model(input_dim=self.test_emb.shape[1], num_classes=num_classes)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def predict(self, output_path: Optional[Path | str] = None) -> Path:
        if output_path is None:
            output_path = self.config.submission_path
        output_path = Path(output_path)

        if self.test_emb is None or len(self.test_emb) == 0:
            logger.info("Preparing embeddings for prediction")
            train_df = self._load_train_df()
            self._load_embeddings(train_df)
            if self.mlb is None:
                self._build_labels(train_df)

        if self.model is None:
            if self.config.model_path.exists():
                self.load(self.config.model_path)
            else:
                raise RuntimeError("Model not trained; call train() first or provide a checkpoint.")

        test_tensor = torch.tensor(self.test_emb, dtype=torch.float32)
        test_loader = DataLoader(TensorDataset(test_tensor), batch_size=self.config.batch_size, shuffle=False)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Writing predictions to %s", output_path)
        with open(output_path, "w", newline="") as f:
            for batch_idx, (inputs,) in enumerate(test_loader):
                inputs = inputs.to(self.device)
                with torch.no_grad():
                    logits = self.model(inputs)
                    probs = torch.sigmoid(logits)
                rows, cols = torch.where(probs > self.config.threshold)
                scores = probs[rows, cols].cpu().numpy()
                rows = rows.cpu().numpy()
                cols = cols.cpu().numpy()
                real_ids = self.test_ids[batch_idx * self.config.batch_size : batch_idx * self.config.batch_size + len(inputs)]
                ids = real_ids[rows]
                terms = self.mlb.classes_[cols]
                for pid, term, score in zip(ids, terms, scores):
                    f.write(f"{pid}\t{term}\t{float(score):.4f}\n")

        return output_path


class ESMModel(EmbeddingMLPModel):
    def __init__(self, **kwargs):
        super().__init__(EmbeddingModelConfig(emb_type="esm", **kwargs))


class T5Model(EmbeddingMLPModel):
    def __init__(self, **kwargs):
        super().__init__(EmbeddingModelConfig(emb_type="prottrans", **kwargs))


class ProtBERTModel(EmbeddingMLPModel):
    def __init__(self, **kwargs):
        super().__init__(EmbeddingModelConfig(emb_type="protbert", **kwargs))
