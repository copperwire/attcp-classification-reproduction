import pytpc
import numpy as np
import scipy as sp
from tqdm import tqdm

repo = "/home/solli-comphys/github/attpc-classification-reproduction/"
data_repo = repo+"data/datapoints/"

import sys

n_samples = 4001

filenames_c = [data_repo+str(1e5+i)+".npy" for i in range(int(10*n_samples))]

filenames_p = [data_repo+str(2e5+i)+".npy" for i in range(int(10*n_samples))]

sys.path.insert(0, repo + "modules")

for i in tqdm(range(0, 10)):
    c_name = repo+"data/C_40000_tilt_largeEvts_{}.h5".format(i)
    p_name = repo+"data/p_40000_tilt_largeEvts_{}.h5".format(i)

    with pytpc.HDFDataFile(c_name, "r") as f:
        from representation_converter import TpcRepresent

        convert_obj = TpcRepresent(filenames_c[i*n_samples: (i+1)*n_samples])
        events = [f[i] for i in range(len(f))]
        convert_obj.convert(events)

    with pytpc.HDFDataFile(p_name, "r") as f:
        from representation_converter import TpcRepresent
        convert_obj = TpcRepresent(filenames_p[i*n_samples: (i+1)*n_samples])
        events = [f[i] for i in range(len(f))]
        convert_obj.convert(events)
