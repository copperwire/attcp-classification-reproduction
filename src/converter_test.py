import pytpc
import numpy as np
import scipy as sp
from tqdm import tqdm

repo = "/home/solli-comphys/github/attpc-classification-reproduction/"
import sys
sys.path.insert(0, repo + "modules")

for i in tqdm(range(1, 10)):
    c_name = repo+"data/C_40000_tilt_largeEvts_{}.h5".format(i)
    p_name = repo+"data/p_40000_tilt_largeEvts_{}.h5".format(i)

    with pytpc.HDFDataFile(c_name, "r") as f:
        from representation_converter import TpcRepresent

        convert_obj = TpcRepresent()
        events = [f[i] for i in range(len(f))]
        A = convert_obj.convert(events)
        sp.sparse.save_npz(
                    repo+"data/C_events_full_{}.npz".format(i),
                    A,
                    compressed=False)

    with pytpc.HDFDataFile(p_name, "r") as f:
        from representation_converter import TpcRepresent
        convert_obj = TpcRepresent()
        events = [f[i] for i in range(len(f))]
        A = convert_obj.convert(events)
        sp.sparse.save_npz(
                    repo+"data/P_events_full{}.npz".format(i),
                    A,
                    compressed=False)
