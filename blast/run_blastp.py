import pandas as pd
from pathlib import Path
from Bio import SeqIO

from profun.models import BlastMatching, BlastConfig
from profun.utils.project_info import ExperimentInfo


data_root = Path('../data/')
train_terms = pd.read_csv(data_root/"Train/train_terms.tsv",sep="\t")

ids = []
seqs = []
with open(data_root/"Train/train_sequences.fasta") as handle:
    for record in SeqIO.parse(handle, "fasta"):
        ids.append(record.id.split("|")[1])
        seqs.append(str(record.seq))
train_seqs_df = pd.DataFrame({'EntryID': ids, 'Seq': seqs})
train_df_long = train_terms.merge(train_seqs_df, on='EntryID')
print(train_df_long.info())

experiment_info = ExperimentInfo(validation_schema='public_lb', 
                                 model_type='blast', model_version='1nn')

config = BlastConfig(experiment_info=experiment_info, 
                    id_col_name='EntryID', 
                    target_col_name='term', 
                    seq_col_name='Seq', 
                    class_names=list(train_df_long['term'].unique()), 
                    optimize_hyperparams=False, 
                    n_calls_hyperparams_opt=None,
                    hyperparam_dimensions=None,
                    per_class_optimization=None,
                    class_weights=None,
                    n_neighbours=5,
                    e_threshold=0.0001,
                    n_jobs=100,
                    pred_batch_size=10
)

blast_model = BlastMatching(config)
blast_model.fit(train_df_long)

test_ids = []
test_seqs = []
with open(data_root/"Test/testsuperset.fasta") as handle:
    for record in SeqIO.parse(handle, "fasta"):
        test_ids.append(record.id)
        test_seqs.append(str(record.seq))
test_seqs_df = pd.DataFrame({'EntryID': test_ids, 'Seq': test_seqs})

# Run full
test_pred_df = blast_model.predict_proba(test_seqs_df.drop_duplicates('EntryID'), return_long_df=True)
test_pred_df.to_csv("blast_pred.tsv", sep="\t", header=False)