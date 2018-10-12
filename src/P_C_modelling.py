import numpy as np
import pandas as pd
import scipy.sparse as sp

repo = "/home/solli-comphys/github/attpc-classification-reproduction/"
A = sp.load_npz(repo+"data/C_events_100.npz")

print(A[0].shape)
