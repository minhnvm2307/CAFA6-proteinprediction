from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Union, Mapping, Optional
from goatools.obo_parser import GODag

import numpy as np
import pandas as pd


def outer_merge_by_weight(
    submission_paths: Iterable[Union[str, Path]],
    threshold: float = 0.01,
    weights: Optional[Mapping[Union[str, Path], float]] = None,
) -> pd.DataFrame:
    """
    Ensemble multiple CAFA submission files by outer merge + weighted sum.

    Each submission file must have format:
        protein <TAB> go_term <TAB> score

    Missing (protein, go_term) in any file is treated as score = 0.
    """
    dfs = []
    paths = [Path(p) for p in submission_paths]

    if weights is None:
        weights = {p: 1.0 for p in paths}

    for i, p in enumerate(paths):
        df = pd.read_csv(
            p,
            sep="\t",
            header=None,
            names=["protein", "go_term", f"score_{i}"]
        )
        dfs.append(df)

    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on=["protein", "go_term"], how="outer")

    score_cols = [c for c in merged.columns if c.startswith("score_")]
    merged[score_cols] = merged[score_cols].fillna(0.0)

    merged["score"] = 0.0
    for i, p in enumerate(paths):
        merged["score"] += weights[p] * merged[f"score_{i}"]

    merged["score"] = merged["score"].clip(0.0, 1.0)
    merged = merged.sort_values("score", ascending=False)
    merged = merged[merged["score"] >= threshold]

    return merged[["protein", "go_term", "score"]]



def load_train_terms(train_terms_path: Path) -> pd.DataFrame:
    df = pd.read_csv(train_terms_path, sep="\t", usecols=["EntryID", "term"])
    df = df.rename(columns={"EntryID": "protein", "term": "go_term"})
    df["score"] = 1.0
    return df

class BranchProtein:
    '''
    Add a prediction to the storage, with optional bonus
    Arguments:
      - protein: Identifier for the protein
      - go_term: GO term that is being predicted
      - score: Confidence score of the prediction
      - branch: Branch of the Gene Ontology (e.g., 'CCO', 'MFO', 'BPO')
      - bonus: Optional bonus to be added to the score
    '''
    
    def __init__(self):
        self.predictions = {}

    def get_submission(self, threshold: float):
        submission = []
        for protein, branches in self.predictions.items():
            for branch, go_terms in branches.items():
                for go_term, score in go_terms:
                    if score >= threshold:
                        submission.append((protein, go_term, score)) 
        return submission


    def add_prediction(self, protein, go_term, score, branch, bonus=0):
        if protein not in self.predictions:
            self.predictions[protein] = {'CCO': {}, 'MFO': {}, 'BPO': {}}
        
        score = float(score)

        if go_term in self.predictions[protein][branch]:
            if self.predictions[protein][branch][go_term] < score:
                self.predictions[protein][branch][go_term] = score + bonus
            else:
                self.predictions[protein][branch][go_term] += bonus
        else:
            self.predictions[protein][branch][go_term] = score

        if self.predictions[protein][branch][go_term] > 1:
            self.predictions[protein][branch][go_term] = 1

    def export_file(self, output_file='submission.tsv', topk=35):
        with open(output_file, 'w') as f:
            for protein, branches in self.predictions.items():
                for branch, go_terms in branches.items():
                    top_go_terms = sorted(go_terms.items(), key=lambda x: x[1], reverse=True)[:topk]

                    for go_term, score in top_go_terms:
                        f.write(f"{protein}\t{go_term}\t{score:.3f}\n")

def extract_go_basic(obo_path):
    """
    Extract GO term -> branch mapping using GOATOOLS.
    Returns: dict
        Mapping: GO_ID -> {'BPO', 'CCO', 'MFO'}
    """
    go_dag = GODag(obo_path, optional_attrs={'namespace'})

    namespace_to_branch = {
        'biological_process': 'BPO',
        'cellular_component': 'CCO',
        'molecular_function': 'MFO',
    }

    go_terms_dict = {}

    for go_id, go_term in go_dag.items():
        if go_term.namespace in namespace_to_branch:
            go_terms_dict[go_id] = namespace_to_branch[go_term.namespace]

    return go_terms_dict