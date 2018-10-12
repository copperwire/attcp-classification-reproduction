import numpy as np
import scipy as sp
import pytpc
import concurrent.futures


class TpcRepresent:
    def __init__(self,):
        self.detector_length = 1250.0
        self.detector_radius = 275.0

    def __grid(self, xyzc, xy_disc=100, z_disc=100):
        dx = 2*self.detector_radius/xy_disc
        dy = 2*self.detector_radius/xy_disc
        dz = self.detector_length/z_disc

        container = np.zeros((z_disc, xy_disc, xy_disc, 1))

        for p in xyzc:
            x_loc = int((p[0] + self.detector_radius)//dx)
            y_loc = int((p[1] + self.detector_radius)//dy)
            z_loc = int(p[2]//dz)

            # print(x_loc, y_loc, z_loc)
            c_norm = p[3]/1e3

            # assert(z_loc > 0 and z_loc < z_disc)
            # assert(x_loc > 0 and x_loc < xy_disc)
            # assert(y_loc > 0 and x_loc < xy_disc)
            container[z_loc, x_loc, y_loc] += c_norm

        return container.flatten("F")

    def wrapper_grid(self, event):
        xyzc = event.xyzs(
                        peaks_only=True,
                        drift_vel=5.2,
                        clock=12.5,
                        return_pads=False,
                        baseline_correction=False,
                        cg_times=False)
        return self.__grid(xyzc)

    def convert(self, file):
        output = list()

        with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
            [output.append(i) for i in executor.map(self.wrapper_grid, file)]

        return sp.sparse.csc_matrix(np.array(output).T)
