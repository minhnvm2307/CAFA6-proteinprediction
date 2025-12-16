# KNN baseline packaged as a CLI wrapper.
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from models.traditional_models import KNNModel, TraditionalConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train or predict the CTD+dipeptide KNN baseline.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--pca-components", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--model-path", type=Path, default=Path("data/checkpoint/knn_model.joblib"))
    parser.add_argument("--submission", type=Path, default=Path("submission_knn.tsv"))
    parser.add_argument("--mode", choices=["train", "predict", "train_and_predict"], default="train_and_predict")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    cfg = TraditionalConfig(
        data_root=args.data_root,
        pca_components=args.pca_components,
        n_neighbors=args.neighbors,
        threshold=args.threshold,
        model_path=args.model_path,
        submission_path=args.submission,
    )
    model = KNNModel(cfg)

    if args.mode in {"train", "train_and_predict"}:
        model.train()
    if args.mode in {"predict", "train_and_predict"}:
        model.predict(args.submission)


if __name__ == "__main__":
    main()
