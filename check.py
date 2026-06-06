import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch, pandas, sklearn
from rdkit import Chem

print("torch", torch.__version__, "| MPS available:", torch.backends.mps.is_available())
print("rdkit ok:", Chem.MolFromSmiles("CCO") is not None)
print("pandas ok:", pandas.read_csv("data/train.csv").shape)