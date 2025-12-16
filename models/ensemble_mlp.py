# Ensemble helper that wraps ensemble/ensemble_model.PredictionEnsemble.
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Sequence

from ensemble.ensemble_model import PredictionEnsemble


class EnsembleModel:
    def __init__(self, go_path: Path, submissions: Iterable[Path], branch_bonus: float = 0.01):
        self.go_path = go_path
        self.submissions = list(submissions)
        self.branch_bonus = branch_bonus

    def branch_topk(self, top_k: int, output_path: Path) -> None:
        pe = PredictionEnsemble(
            go_train_path=str(self.go_path),
            submission_paths=[str(p) for p in self.submissions],
            branch_bonus=self.branch_bonus,
        )
        pe.topk_ensemble(top_k=top_k, output_path=output_path)

    def weighted(self, weights, threshold: float, output_path: Path):
        pe = PredictionEnsemble(
            go_train_path=str(self.go_path),
            submission_paths=[str(p) for p in self.submissions],
            branch_bonus=self.branch_bonus,
        )
        pe.weighted_ensemble(weights=weights, threshold=threshold, output_path=output_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Branch-based ensemble of submission files.")
    parser.add_argument("--go-path", type=Path, default=Path("data/Train/go-basic.obo"))
    parser.add_argument(
        "--submissions",
        type=Path,
        nargs="+",
        default=[
            Path("data/checkpoint/submissions/submission_t5.tsv"),
            Path("data/checkpoint/submissions/submission_blast.tsv"),
        ],
    )
    parser.add_argument("--branch-bonus", type=float, default=0.01)
    parser.add_argument("--top-k", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("ensemble_submission.tsv"))
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    model = EnsembleModel(args.go_path, args.submissions, branch_bonus=args.branch_bonus)
    model.branch_topk(top_k=args.top_k, output_path=args.output)


if __name__ == "__main__":
    main()
