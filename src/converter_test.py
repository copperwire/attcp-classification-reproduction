import pytpc
import numpy as np
import scipy as sp

repo = "/home/solli-comphys/github/attpc-classification-reproduction/"
import sys
sys.path.insert(0, repo + "modules")


with pytpc.HDFDataFile(repo+"/data/C_40000_tilt_largeEvts.h5", "r") as f:
    from representation_converter import TpcRepresent

    i = 0
    inc = 100
    while i < 4000:
        convert_obj = TpcRepresent()
        events = [f[i] for i in range(i, i+inc-1)]
        A = convert_obj.convert(events)
        sp.sparse.save_npz(
                    repo+"data/C_events_{:}.npz".format(i),
                    A,
                    compressed=False)
        i += inc


with pytpc.HDFDataFile(repo+"/data/p_40000_tilt_largeEvts.h5", "r") as f:
    from representation_converter import TpcRepresent

    i = 0
    inc = 100
    while i < 4000:
        convert_obj = TpcRepresent()
        events = [f[i] for i in range(i, i+inc-1)]
        A = convert_obj.convert(events)
        sp.sparse.save_npz(
                    repo+"data/P_events_{:}.npz".format(i),
                    A,
                    compressed=False)
        i += inc
