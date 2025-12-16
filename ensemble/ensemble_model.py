from pathlib import Path
from typing import Dict, Iterable, List, Optional
from tqdm import tqdm

import pandas as pd
from ensemble.ensemble_utils import (
    BranchProtein,
    extract_go_basic,
    load_train_terms,
    outer_merge_by_weight
)


class PredictionEnsemble:
    def __init__(self, go_train_path: str, submission_paths: Iterable[Path | str], branch_bonus: float = 0.01):
        self.go_branch_map = extract_go_basic(go_train_path)
        self.branch_bonus = branch_bonus
        self.submission_paths = submission_paths

    def topk_ensemble(self, top_k: int = 50, output_path: Path | str | None = None,
    ) -> pd.DataFrame:
        '''
        Ensemble results by merge multiple predictions without weights
        With each protein take topK go-terms
        '''
        predictor = BranchProtein()
        for path in self.submission_paths:
            for item in tqdm(open(path)):
                item_list = item.split('\t')
                protein_id = item_list[0]
                go_term=item_list[1].strip()
                score = float(item_list[2].strip())
                if go_term in self.go_branch_map:
                    root = self.go_branch_map[go_term]
                    predictor.add_prediction(protein_id, go_term, score, root, bonus=self.branch_bonus)
        
        predictor.export_file(output_path, topk=top_k)

    def weighted_ensemble(
        self,
        weights: Dict[str, float],
        threshold: float = 0.01,
        output_path: Path | str | None = None,
    ) -> pd.DataFrame:
        '''
        Ensemble results by merge multiple predictions with weights
        With each protein take topK go-terms
        '''
        ensemble_df = outer_merge_by_weight(self.submission_paths, threshold=threshold, weights=weights)
        ensemble_df.to_csv(output_path, sep="\t", header=False, index=False)
        return ensemble_df

    def merge_with_train(
        self,
        train_terms_path: Path | str,
        threshold: float = 0.01,
        output_path: Path | str | None = None,
    ) -> pd.DataFrame:
        '''
        Ensemble results by merge multiple predictions without weights
        Apply trainset to increase correction
        '''
        predictor = BranchProtein()
        for path in self.submission_paths:
            for item in tqdm(open(path)):
                item_list = item.split('\t')
                protein_id = item_list[0]
                go_term=item_list[1].strip()
                score = float(item_list[2].strip())
                if go_term in self.go_branch_map:
                    root = self.go_branch_map[go_term]
                    predictor.add_prediction(protein_id, go_term, score, root, bonus=self.branch_bonus)

        merged = pd.DataFrame(predictor.get_submission(threshold))
        train_df = load_train_terms(Path(train_terms_path))
        df = pd.concat([merged, train_df], ignore_index=True)
        df = df.groupby(["protein", "go_term"], as_index=False)["score"].max()
        df.to_csv(output_path, sep="\t", header=False, index=False)

        return df
