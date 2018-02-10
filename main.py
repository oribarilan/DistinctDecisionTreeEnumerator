import pandas as pd
import numpy as np
from dtunique import DTdistinct

############################  MAIN  ##############################

dir_ = "/Users/ori/Documents/vscode-docs/ML/proj/datasets/"

# V1 & V2 are zero-based list of indexes
datasets = [
    # {
    #     "name" : "Adult",
    #     "path" : dir_ + "adult.csv",
    #     "delimiter" : ',',
    # },
    {
        "name" : "Flags",
        "path" : dir_ + "flags_formatted.csv",
        "delimiter" : ',',
    }
]

for ds_dict in datasets:
    print("starting {0}".format(ds_dict["name"]))
    dataset = pd.read_csv(ds_dict["path"])
    dtd = DTdistinct(dataset, random_state=7)
    S = set(range(dataset.shape[1]-1))
    R = set()
    trees = dtd.DTdistinct_enumerator(R, S)
    print(len(trees))