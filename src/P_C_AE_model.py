repo = "/home/solli-comphys/github/attpc-classification-reproduction/"
import numpy as np
from keras.models import load_model
from sklearn.model_selection import train_test_split
import concurrent.futures

import sys
sys.path.insert(0, repo+"modules")

autoencoder = load_model(repo+"models/autoencoder.h5")

# %%
n_samples = 10*4001
sample_dim = 20*20*20
num_training_samples = 0.7*4001
num_validation_samples = 0.3*4001
batch_size = 1e2


data_path = repo+"data/datapoints/"
fnames_c = [data_path + str(1e5 + i) + ".npy"
            for i in range(int(n_samples))]
targets_c = [0 for io in range(int(n_samples))]

fnames_p = [data_path + str(2e5 + i) + ".npy"
            for i in range(int(n_samples))]


targets_p = [1 for i in range(int(n_samples))]

fnames = np.array(fnames_c+fnames_p)
targets = np.array(targets_c+targets_p)
# %%
from keras.models import Model
from keras import backend as K


def file_load_wrapper(tup):
    return np.load(tup[0]), tup[1]


sample = np.zeros((len(fnames), sample_dim))
target_ind = np.zeros(len(targets))
ft_iter = ((f, i) for f, i in zip(fnames, targets))

with concurrent.futures.ProcessPoolExecutor(
                                            max_workers=3,
                                            ) as executor:
    for i, tup in enumerate(executor.map(file_load_wrapper, ft_iter)):
        sample[i] = tup[0]
        target_ind[i] = tup[1]


# %%
encoder_model = Model(
                    inputs=autoencoder.layers[0].input,
                    outputs=autoencoder.layers[1].output
                    )
# encoder_model.compile(loss="mean_squared_error", optimizer="adam")
# %%

data = encoder_model.predict(sample)
np.save(repo+"data/processed/enc_full_20.npy", data)


# %%

np.save(repo+"data/processed/targets.npy", target_ind)
