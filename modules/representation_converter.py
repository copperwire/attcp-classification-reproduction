import numpy as np
import scipy as sp
import pytpc
import concurrent.futures


class TpcRepresent:
    def __init__(
        self,
        unique_map
            ):

        self.detector_length = 1250.0
        self.detector_radius = 275.0
        self.unique_map = unique_map

    def __grid(self, xyzc, xy_disc=20, z_disc=20):
        dx = 2*self.detector_radius/xy_disc
        dy = 2*self.detector_radius/xy_disc
        dz = self.detector_length/z_disc
        input_shape = xyzc.shape

        container = np.zeros((z_disc, xy_disc, xy_disc, 1))

        x_loc = ((xyzc[:, 0] + self.detector_radius)//dx).astype(int)
        y_loc = ((xyzc[:, 1] + self.detector_radius)//dy).astype(int)
        z_loc = (xyzc[:, 2]//dz).astype(int)

        # print(x_loc, y_loc, z_loc)
        c_norm = (xyzc[:, 3]/1e3).reshape((input_shape[0], 1))

        # assert(z_loc > 0 and z_loc < z_disc)
        # assert(x_loc > 0 and x_loc < xy_disc)
        # assert(y_loc > 0 and x_loc < xy_disc)

        container[z_loc, x_loc, y_loc] += c_norm

        ret_obj = container.flatten("F")
        return ret_obj


    def wrapper_grid(self, event):
        xyzc = event.xyzs(
                        peaks_only=True,
                        drift_vel=5.2,
                        clock=12.5,
                        return_pads=False,
                        baseline_correction=False,
                        cg_times=False)
        return self.__grid(xyzc)

    def writer_func(self, coord, fname):
        np.save(fname, coord)
        return 0

    def convert(self, file):
        output = list()

        with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
            [output.append(i) for i in executor.map(self.wrapper_grid, file)]

        with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
            [i for i in executor.map(self.writer_func, output, self.unique_map)]
