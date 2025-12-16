from pathlib import Path

from ensemble.ensemble_model import PredictionEnsemble


if __name__ == "__main__":
    default_dir = Path("data/checkpoint/submissions")
    submission_files = [
        default_dir / "submission_t5.tsv",
        # default_dir / "submission_esm.tsv",
        # default_dir / "submission_protbert.tsv",
        # default_dir / "submission_knn.tsv",
        default_dir / "submission_blast.tsv",
    ]

    go_path = "data/Train/go-basic.obo"
    pe = PredictionEnsemble(go_train_path=go_path, submission_paths=submission_files, branch_bonus=0.01)

    print(f"Ensembling {len(submission_files)} submissions with Branch method")
    out_file = Path("ensemble_submission.tsv")
    df = pe.topk_ensemble(top_k=7, output_path=out_file)
    if out_file.exists():
        print(f"Created submission file to {out_file}")
    else:
        print(df.info())
        print(df.head(3))
