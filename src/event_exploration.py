from pytpc.hdfdata import HDFDataFile
import pytpc

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt


repo = "/home/solli-comphys/github/attpc-classification-reproduction/"

evt = 0

with pytpc.HDFDataFile(repo+"/data/C_40000_tilt_largeEvts.h5", "r") as f:
    evt = f[3]

space_distr = evt.xyzs(
                peaks_only=True,
                drift_vel=5.2,
                clock=12.5,
                return_pads=False,
                baseline_correction=False,
                cg_times=False)

xs = space_distr[:, 0]
ys = space_distr[:, 1]
zs = space_distr[:, 2]
ch = space_distr[:, 3]

import pandas as pd

charge_series = pd.DataFrame(ch)


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(xs, ys, zs, c=ch, s=20, cmap="gray")
plt.show()
