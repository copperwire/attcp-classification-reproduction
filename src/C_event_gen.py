import math
#import numpy as np

import os

import pytpc
import yaml
#import h5py

from effsim.paramgen import uniform_param_generator
from effsim.paramgen import distribution_param_generator
from effsim.effsim import EventSimulator, NoiseMaker
from pytpc.hdfdata import HDFDataFile
from tqdm import tqdm

os.chdir("/home/solli-comphys/github/attpc-classification-reproduction/")
repo="/home/solli-comphys/github/attpc-classification-reproduction/"

with open("pytpc_config/config_e15503b_C.yml") as f:
    config = yaml.load(f)

# the generation of events stolen in their entirety from github.com/jtaylorz

beam_enu0 = config['beam_enu0']
beam_mass = config['beam_mass']
beam_charge = config['beam_charge']
mass_num = config['mass_num']
max_beam_angle = (config['max_beam_angle']*math.pi)/180
beam_origin_z = config['beam_origin_z']

gas = pytpc.gases.InterpolatedGas('isobutane', 19.2)

POINT_CUTOFF = 30

# number of events to create
num_evts = 4000

# doubling events generated to cushion for possibility
# of failed event sim and small events

for i in tqdm(range(1, 10)):
    Cgen = uniform_param_generator(
                beam_enu0,
                beam_mass,
                beam_charge,
                mass_num,
                max_beam_angle,
                beam_origin_z,
                gas,
                num_evts*2
                )

    sim = EventSimulator(config)

    fname = repo + '/data/C_40000_tilt_largeEvts_{}.h5'.format(i)
    with HDFDataFile(fname, 'w') as hdf:
        evt_id = 0
        for C in Cgen:
            if(evt_id > num_evts):
                break
            else:
                try:
                    evt, ctr = sim.make_event(
                                C[0][0],
                                C[0][1],
                                C[0][2],
                                C[0][3],
                                C[0][4],
                                C[0][5]
                                )

                except IndexError:
                    print("Bad event, skipping")
                    continue

            pyevt = sim.convert_event(evt, evt_id)

            cur_ptcount = pyevt.xyzs(
                            peaks_only=True,
                            drift_vel=5.2,
                            clock=12.5,
                            return_pads=False,
                            baseline_correction=False,
                            cg_times=False).shape[0]

            if(cur_ptcount < POINT_CUTOFF):
                print("Event with low point count, skipping")
                continue
            else:
                hdf.write_get_event(pyevt)
                print(
                        "Wrote event " +
                        str(evt_id) +
                        " with " +
                        str(len(pyevt.traces)) +
                        " traces")

                evt_id += 1

    print(str(evt_id-1) + " events written to file")
