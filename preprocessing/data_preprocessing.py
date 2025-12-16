import numpy as np
import pandas as pd
from Bio import SeqIO
from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from typing import Callable, Dict, Optional, Tuple, Any

from .utils import (
    create_seq_df, 
    create_train_df, 
    create_term_df, 
    create_dataset_with_emb,
    remove_seq_outliers,
    remove_term_by_freq,
)



class ProteinDataProcessor:
    """
    Data preprocessing helper that can dynamically apply or skip filtering steps.

    Built-in sequence filters:
        - "outlier": drop sequences above a percentile length.
        - "none": keep sequences unchanged.

    Built-in term filters:
        - "frequency": drop terms below a frequency threshold.
        - "none": keep terms unchanged.
    """

    def __init__(self, data_folder: str, seq_type: str = "list"):
        self.data_folder = data_folder
        self.seq_type = seq_type

        # Registry for filter strategies
        self.seq_filters: Dict[str, Callable[[pd.DataFrame, Any], Tuple[pd.DataFrame, Dict[str, Any]]]] = {
            "outlier": self._filter_sequence_outliers,
            "none": lambda df, **_: (df, {"strategy": "none", "dropped": 0}),
        }
        self.term_filters: Dict[str, Callable[[pd.DataFrame, Any], Tuple[pd.DataFrame, Dict[str, Any]]]] = {
            "frequency": self._filter_term_frequency,
            "none": lambda df, **_: (df, {"strategy": "none", "dropped": 0}),
        }

    def load_sequences(self) -> pd.DataFrame:
        return create_seq_df(self.data_folder, seq_type=self.seq_type)

    def load_terms(self) -> pd.DataFrame:
        return create_term_df(self.data_folder)

    def _filter_sequence_outliers(
        self, seq_df: pd.DataFrame, prob_keep: int = 99, **kwargs
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        filtered_df, cutoff = remove_seq_outliers(seq_df, prob_keep=prob_keep, return_cutoff=True)
        dropped = len(seq_df) - len(filtered_df)
        return filtered_df, {
            "strategy": "outlier",
            "prob_keep": prob_keep,
            "cutoff_length": cutoff,
            "dropped": dropped,
        }

    def _filter_term_frequency(
        self, term_df: pd.DataFrame, freq_threshold: int = 10, **kwargs
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        filtered_df, stats = remove_term_by_freq(term_df, freq_threshold=freq_threshold, return_stats=True)
        stats.update({"strategy": "frequency"})
        return filtered_df, stats

    @staticmethod
    def _apply_filter(
        df: pd.DataFrame,
        strategy: Optional[str | Callable],
        registry: Dict[str, Callable],
        kwargs: Optional[Dict[str, Any]],
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if strategy is None or strategy == "none":
            return df, {"strategy": "none", "dropped": 0}

        if callable(strategy):
            result = strategy(df, **(kwargs or {}))
            if isinstance(result, tuple) and len(result) == 2:
                return result  # user provided (df, info)
            return result, {"strategy": getattr(strategy, "__name__", "callable")}

        if strategy not in registry:
            raise ValueError(f"Unknown filter strategy '{strategy}'")

        return registry[strategy](df, **(kwargs or {}))

    def prepare_training_data(
        self,
        seq_filter: Optional[str | Callable] = "outlier",
        term_filter: Optional[str | Callable] = "frequency",
        seq_filter_kwargs: Optional[Dict[str, Any]] = None,
        term_filter_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
        """
        Load, optionally filter, and merge training data.

        Args:
            seq_filter: strategy name or callable for sequence filtering; None or "none" keeps original.
            term_filter: strategy name or callable for term filtering; None or "none" keeps original.
            seq_filter_kwargs: extra params passed to the sequence filter.
            term_filter_kwargs: extra params passed to the term filter.

        Returns:
            train_df: merged dataframe of sequences and terms.
            filter_info: metadata describing applied filters.
        """
        seq_df = self.load_sequences()
        term_df = self.load_terms()

        seq_df, seq_info = self._apply_filter(seq_df, seq_filter, self.seq_filters, seq_filter_kwargs)
        term_df, term_info = self._apply_filter(term_df, term_filter, self.term_filters, term_filter_kwargs)

        train_df = create_train_df(seq_df, term_df)
        return train_df, {"sequence": seq_info, "term": term_info}

if __name__ == "__main__":
    train_path = './data'
    processor = ProteinDataProcessor(train_path, 'str')
    train_df, info = processor.prepare_training_data()

    print(train_df.head())
    print(info)

    print("=====Load embedding for train and test set======")
    sample_train_ids = ["A0A0C5B5G6", "A0JNW5", "A0JP26"]
    sample_test_ids = ["A0JNW5", "A0A0C5B5G6", "A1A4S6", "A1A519"]

    train_emb, test_emb = create_dataset_with_emb(sample_train_ids, sample_test_ids, "esm")
    print("Load embeddings with:", "\n- Training shape:", np.array(train_emb).shape, "\n- Testing shape", np.array(test_emb).shape)
