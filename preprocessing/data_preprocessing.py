import numpy as np
import pandas as pd
from Bio import SeqIO
from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from typing import Callable, Dict, Optional, Tuple, Any

def create_seq_df(data_folder: str, seq_type: str = "list"):
    train_seq_path = f'{data_folder}/Train/train_sequences.fasta'
    print(f"Read {train_seq_path}")
    train_seq = pd.DataFrame([
        {"ID": record.id.split("|")[1], "Sequence": list(record.seq) if seq_type=="list" else str(record.seq)}
        for record in SeqIO.parse(train_seq_path, format='fasta')
    ])

    print(f"Loaded Training sequence size: {len(train_seq)}")

    return train_seq

def create_term_df(data_folder):
    term_path = f'{data_folder}/Train/train_terms.tsv'
    print(f'Read {term_path}')

    train_term_df = pd.read_csv(term_path, sep='\t')
    train_term = train_term_df.groupby("EntryID", as_index=False)['term'].apply(list)

    print(f"Loaded Training sequence size: {len(train_term)}")

    return train_term

def create_train_df(seq_df: pd.DataFrame, term_df: pd.DataFrame):
    train_df = seq_df.merge(term_df, how="inner", left_on="ID", right_on="EntryID", right_index=False)
    train_df = train_df.drop(columns=["EntryID"])

    if train_df.isnull().values.any():
        print("WARNING: Data contains NaN values after merging!")
    
    print("Training dataframe size:", len(train_df))
    print(train_df.head(3))
    
    return train_df

"""
Data Analysis
1. Sequence Length Distribution - Remove statistical outliers
2. Number of GO term / Protein - Filter term with presenting threshold
"""
def seq_len_dist(seq_df: pd.DataFrame):
    seq_lengths = [len(seq) for seq in seq_df["Sequence"].to_list()]
    sns.histplot(seq_lengths, bins=50)
    plt.xlabel("Sequence Length")
    plt.ylabel("Count")
    plt.title("Distribution of Protein Sequence Lengths")
    plt.show()

def remove_seq_outliers(seq_df: pd.DataFrame, prob_keep: int = 99, *, return_cutoff: bool = False):
    seq_lengths = seq_df["Sequence"].apply(len)
    cutoff = np.percentile(seq_lengths, prob_keep)
    
    print(f"Removing sequences longer than {cutoff} (Top {100-prob_keep}%)")

    filtered_seq_df = seq_df[seq_lengths < cutoff].copy()

    sns.histplot(filtered_seq_df["Sequence"].apply(len), bins=100)
    plt.xlabel(f"Sequence Length (< {prob_keep}% quantile)")
    plt.ylabel("Count")
    plt.title("Protein Sequence Lengths (Outliers Removed)")
    plt.show()

    print(f"Original size: {len(seq_df)}, New size: {len(filtered_seq_df)}")

    if return_cutoff:
        return filtered_seq_df, cutoff
    return filtered_seq_df

def term_freq_dist(term_df: pd.DataFrame):
    # Number of terms that a protein have in dataset
    term_per_protein = term_df["term"].apply(len)

    allterms = [t for sub_term in term_df["term"] for t in sub_term]
    cnt = Counter(allterms)
    
    term_freq = list(cnt.values())
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.histplot(term_per_protein, kde=True, bins=50, ax=axes[0])
    axes[0].set_xlabel("Number of GO terms per protein")
    axes[0].set_ylabel("Count")
    axes[0].set_title("GO Terms per Protein Distribution")

    sns.histplot(term_freq, bins=100, ax=axes[1], log_scale=True)
    axes[1].set_xlabel("Frequency of GO term")
    axes[1].set_ylabel("Count")
    axes[1].set_title("GO Term Frequency Distribution")

    plt.tight_layout()
    plt.show()


def remove_term_by_freq(term_df: pd.DataFrame, freq_threshold: int = 10, *, return_stats: bool = False):
    print(f"--- Filtering terms with frequency <= {freq_threshold} ---")

    allterms = [t for sub_term in term_df["term"] for t in sub_term]
    cnt = Counter(allterms)

    valid_terms = set([l for l,count in cnt.items() if count > freq_threshold])
    print("Unique terms decrease from:", len(cnt), "to", len(valid_terms))  

    # Filtering
    def filter_row(term_list):
        return [t for t in term_list if t in valid_terms]

    filtered_term = term_df.copy()
    filtered_term["term"] = filtered_term["term"].apply(filter_row)

    n_original_proteins = len(filtered_term)
    filtered_term = filtered_term[filtered_term['term'].map(len) > 0]
    n_remaining_proteins = len(filtered_term)

    print(f"Removed {n_original_proteins - n_remaining_proteins} proteins that became empty after filtering.")
    print(f"Final DataFrame shape: {filtered_term.shape}")

    stats = {
        "dropped_proteins": n_original_proteins - n_remaining_proteins,
        "kept_terms": len(valid_terms),
        "total_terms": len(cnt),
        "freq_threshold": freq_threshold,
    }

    if return_stats:
        return filtered_term, stats
    return filtered_term

def create_dataset_with_emb(
    train_ids: list,
    test_ids: list,
    emb_type: str,
    embedding_root: Optional[str | Path] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load embeddings for the provided train/test IDs from the chosen embedding set.

    Embedding folders (protbert, esm, prottrans) contain two files:
        - test_ids.npy
        - test_embeddings.npy
    These files store embeddings for both the training and test proteins; this
    function slices out the rows corresponding to the supplied IDs.

    Args:
        train_ids: Protein IDs for the training split.
        test_ids: Protein IDs for the test split.
        emb_type: One of {"protbert", "esm", "prottrans"} selecting the folder.
        embedding_root: Optional custom root directory containing the embedding
            subfolders; if not provided, default search paths under ./data are used.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (train_embeddings, test_embeddings) aligned
        with the order of train_ids and test_ids respectively.
    """
    emb_type = emb_type.lower()
    valid_types = {"protbert", "esm", "prottrans"}
    if emb_type not in valid_types:
        raise ValueError(f"emb_type must be one of {valid_types}, got '{emb_type}'.")

    base_dir = Path(__file__).resolve().parent.parent / "data" / "embedding" / emb_type
    ids_path = base_dir / "test_ids.npy"
    emb_path = base_dir / "test_embeddings.npy"

    if ids_path is None or emb_path is None:
        raise FileNotFoundError(
            f"Embedding files not found. Looked in: {base_dir}. "
            "Expected test_ids.npy and test_embeddings.npy."
        )

    ids = np.load(ids_path, allow_pickle=True)
    embeddings = np.load(emb_path)

    if len(ids) != len(embeddings):
        raise ValueError(f"IDs and embeddings have mismatched lengths: {len(ids)} vs {len(embeddings)}.")

    embedding_map = {str(pid): emb for pid, emb in zip(ids, embeddings)}

    def _extract(target_ids: list, split_name: str):
        missing = [pid for pid in target_ids if str(pid) not in embedding_map]
        if missing:
            preview = ", ".join(map(str, missing[:5]))
            raise KeyError(f"Missing embeddings for {len(missing)} {split_name} IDs. First few: {preview}")
        return np.stack([embedding_map[str(pid)] for pid in target_ids])

    train_emb = _extract(train_ids, "train")
    test_emb = _extract(test_ids, "test")

    return train_emb, test_emb



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
    train_path = '../data/'
    processor = ProteinDataProcessor(train_path, 'str')
    train_df, info = processor.prepare_training_data()

    print(train_df.head())
    print(info)

    print("=====Load embedding for train and test set======")
    sample_train_ids = ["A0A0C5B5G6", "A0JNW5", "A0JP26"]
    sample_test_ids = ["A0JNW5", "A0A0C5B5G6", "A1A4S6", "A1A519"]

    train_emb, test_emb = create_dataset_with_emb(sample_train_ids, sample_test_ids, "prottrans")
    print("Load embeddings with:", "\n- Training shape:", np.array(train_emb).shape, "\n- Testing shape", np.array(test_emb).shape)
