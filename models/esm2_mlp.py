# Training/prediction entrypoint for the ESM2 embedding MLP.
from __future__ import annotations

import argparse
from pathlib import Path

import logging
from models.embedding_models import ESMModel


def parse_args():
    parser = argparse.ArgumentParser(description="Train or predict using the ESM2 MLP model.")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Path to data root (contains Train/, embedding/).")
    parser.add_argument("--model-path", type=Path, default=Path("data/checkpoint/esm_model.pth"), help="Where to save/load weights.")
    parser.add_argument("--submission", type=Path, default=Path("submission_esm.tsv"), help="Where to write predictions.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--mode", choices=["train", "predict", "train_and_predict"], default="train_and_predict")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    model = ESMModel(
        data_root=args.data_root,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        threshold=args.threshold,
        model_path=args.model_path,
        submission_path=args.submission,
    )

    if args.mode in {"train", "train_and_predict"}:
        model.train()
    if args.mode in {"predict", "train_and_predict"}:
        model.predict(args.submission)


if __name__ == "__main__":
    main()
