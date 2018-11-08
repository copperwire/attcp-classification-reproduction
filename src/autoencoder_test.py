repo = "/home/solli-comphys/github/attpc-classification-reproduction/"
import numpy as np
import keras as kr
import tensorflow as tf
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, repo+"modules")

#tf.enable_eager_execution()

n_samples = 10*4001
sample_dim = 20*20*20
batch_size = int(1e3)
num_epochs = 5
num_training_samples = 0.7*4001
num_validation_samples = 0.3*4001

data_path = repo+"data/datapoints/"
fnames_c = [data_path + str(1e5 + i) + ".npy"
            for i in range(int(n_samples))]
targets_c = [0 for io in range(int(n_samples))]

fnames_p = [data_path + str(2e5 + i) + ".npy"
            for i in range(int(n_samples))]


targets_p = [1 for i in range(int(n_samples))]

fnames = np.array(fnames_c+fnames_p)
targets = np.array(targets_c+targets_p)

fnames_c, fnames_p = None, None
targets_p, targets_c = None, None

X_train, X_test, y_train, y_test = train_test_split(
                                        fnames,
                                        targets,
                                        test_size=0.3,
                                        random_state=42,
                                        shuffle=True
                                        )

# %%
from sample_generator import SampleSequence

training_batch_gen = SampleSequence(X_train, y_train, batch_size)
test_batch_gen = SampleSequence(X_test, y_test, batch_size)

a = training_batch_gen[0]
# %%
import matplotlib.pyplot as plt

a = np.linspace(1e-6, 1, 100)
b = training_batch_gen.distr(a, 2)
plt.plot(a, b)

# %%
from keras.layers import Input, Dense
from keras.models import Model

# this is the size of our encoded representations
encoding_dim = 800

# this is our input placeholder
input_rep = Input(shape=(20*20*20,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_rep)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(8000, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_rep, decoded)
autoencoder.compile(optimizer="sgd", loss="mean_squared_error")


autoencoder.fit_generator(
                generator=training_batch_gen,
                steps_per_epoch=(num_training_samples//batch_size),
                epochs=num_epochs,
                verbose=1,
                validation_data=test_batch_gen,
                validation_steps=(num_validation_samples//batch_size),
                use_multiprocessing=True,
                workers=6,
                max_queue_size=10
                )


autoencoder.save(repo+"models/autoencoder.h5")
# %%
