import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score

import matplotlib.pyplot as plt

repo = "/home/solli-comphys/github/attpc-classification-reproduction/"

# %%
avg_nonzero = []

means_c = []
means_p = []

stds_c  = []
stds_p = []

maxs_c = []
maxs_p = []

mins_c =  []
mins_p = []



import seaborn as sns

p_fig, p_ax = plt.subplots()
c_fig, c_ax = plt.subplots()

for i in range(1, 10):
    tmp = sp.load_npz(repo+"data/C_events_full_{}.npz".format(i))
    full_C = tmp.todense().T

    tmp = sp.load_npz(repo+"data/P_events_full{}.npz".format(i))
    full_P = tmp.todense().T

    nzc = np.array(full_C[np.nonzero(full_C)])
    nzp = np.array(full_P[np.nonzero(full_P)])

    sns.distplot(nzc, ax=c_ax, kde=False, norm_hist=True)
    sns.distplot(nzp, ax=p_ax, kde=False, norm_hist=True)

    means_c.append(np.mean(nzc))
    means_p.append(np.mean(nzp))

    stds_c.append(nzc.std())
    stds_p.append(nzp.std())


    maxs_c.append(nzc.max())
    maxs_p.append(nzp.max())


    mins_c.append(nzc.min())
    mins_p.append(nzp.min())

# %%
p_ax.set_xlim((0, 2))
p_fig
# %%

data = np.load(repo+"data/processed/enc_full_20.npy")
targets = np.load(repo+"data/processed/targets.npy")

X_train, X_test, y_train, y_test = train_test_split(
        data, targets, test_size=0.33, random_state=42, shuffle=True)

L2 = [1e-3, 1e-2, 1e-1, 1e0]
fig, ax = plt.subplots()

for L in L2:
    width = 50
    depth = 3
    topology = (width, )*depth

    lr_model = MLPClassifier(
                    hidden_layer_sizes=topology,
                    alpha=L,
                    learning_rate="adaptive"
                    )
    lr_model.fit(X_train, y_train)

    decision_function = lr_model.predict_proba(X_test)[:,1]
    fpr, tpr,  _ = roc_curve(y_test, decision_function)
    auc = roc_auc_score(y_test, decision_function)
    acc = accuracy_score(y_test, lr_model.predict(X_test))

    ax.plot(fpr, tpr,
            label=r"$\lambda = $ = {:.2e}  AUC={:.2f}  ACC={:.3f}".format(L, auc, acc))

plt.legend()
plt.show()
