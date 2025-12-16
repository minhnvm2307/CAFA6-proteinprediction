
# Data preprocessing overview

This folder prepares the CAFA training data before modeling.

- **Inputs**: `Train/train_sequences.fasta` and `Train/train_terms.tsv` under the configured data root.
- **Sequence filtering**: Optional length-outlier removal (keeps sequences under a chosen percentile cutoff).
- **Term filtering**: Optional pruning of rare GO terms below a frequency threshold; proteins that lose all labels are removed.
- **Outputs**: A cleaned training `DataFrame` with `ID`, `Sequence`, and `term` columns, plus metadata about applied filters.

To run programmatically, use `ProteinDataProcessor` in `data_preprocessing.py`:

```python
from preprocessing.data_preprocessing import ProteinDataProcessor

processor = ProteinDataProcessor("../data", seq_type="list")
train_df, filter_info = processor.prepare_training_data(
    seq_filter="outlier",                 # or "none"
    term_filter="frequency",              # or "none"
    seq_filter_kwargs={"prob_keep": 98},  # optional
    term_filter_kwargs={"freq_threshold": 20},
)
```

## Sample ouput
```
Original size: 82404, New size: 81578
--- Filtering terms with frequency <= 10 ---
Unique terms decrease from: 26125 to 7257
Removed 1501 proteins that became empty after filtering.
Final DataFrame shape: (80903, 2)
Training dataframe size: 80089
    ID                  Sequence                    term                               
0  A0A0C5B5G6   MRWQEMGYIFYPRKLR           [GO:0001649, GO:0033687, GO:0005615, GO:000563...
1  A0JNW5   MAGIIKKQILKHLSRFTKNLSPDKINLSTLKGEGELKNLELDEEVL...  [GO:0120013, GO:0034498, GO:0005769, GO:012000...
2  A0JP26   MVAEVCSMPAASAVKKPFDLRSKMGKWCHHRFPCCRGSGKSNMGTS...                                       [GO:0005515]

{
    'sequence': {
        'strategy': 'outlier', 'prob_keep': 99, 'cutoff_length': np.float64(2375.0), 'dropped': 826
    }, 

    'term': {
        'dropped_proteins': 1501, 'kept_terms': 7257, 'total_terms': 26125, 'freq_threshold': 10, 'strategy': 'frequency'
    }
}
```

## Get embedding for training set and testing set

**Funtion**: create_dataset_with_emb

```
    train_ids: list,
    test_ids: list,
    emb_type: str,
    embedding_root: Optional[str | Path] = None,
```
- Load embedding of train/test protein ids with 3 type of emb model `["esm", "prottrans", "protbert"]`


```Python
train_emb, test_emb = create_dataset_with_emb(sample_train_ids, sample_test_ids, "prottrans")

sample_train_ids = ["A0A0C5B5G6", "A0JNW5", "A0JP26"]
sample_test_ids = ["A0JNW5", "A0A0C5B5G6", "A1A4S6", "A1A519"]

print("Load embeddings with:", "\n- Training shape:", np.array(train_emb).shape, "\n- Testing shape", np.array(test_emb).shape)

# Output
=====Load embedding for train and test set======
Load embeddings with: 
- Training shape: (3, 1024) 
- Testing shape (4, 1024)
```



