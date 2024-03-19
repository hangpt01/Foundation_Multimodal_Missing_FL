import torch
import pyarrow as pa
import numpy as np
from collections import Counter

for mode in ['train','test']:
    tables = [
                    pa.ipc.RecordBatchFileReader(
                    pa.memory_map(f"benchmark/RAW_DATA/FOOD101/food101_{mode}.arrow", "r")
                ).read_all()
            ]

    remove_duplicate = False
    table = pa.concat_tables(tables, promote=True)

    #-------------------------------------------------------
    # use a subset of data 
    total_rows = table.num_rows
    # Determine the range of rows you want to extract (for example, the first quarter of the data)
    start_index = 0
    end_index = total_rows // 10.04
    # Extract the subset of the table
    table = table.slice(start_index, end_index)
    #-------------------------------------------------------

    # all_texts = table['text'].to_pandas().tolist()
    # all_texts = (      # len: 61227
    #     [list(set(texts)) for texts in all_texts]
    #     if remove_duplicate
    #     else all_texts
    # )

    labels = table["label"].to_pandas().tolist()
    label_np = np.array(labels)
    print(mode)
    print(label_np.shape, np.unique(label_np))
    print(Counter(labels))
# import pdb; pdb.set_trace()