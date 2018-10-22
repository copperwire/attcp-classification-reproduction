import numpy as np
import scipy.sparse as sp

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score

import matplotlib.pyplot as plt

repo = "/home/solli-comphys/github/attpc-classification-reproduction/"

tmp = sp.load_npz(repo+"data/C_events_full.npz")
full_C = tmp.todense().T

tmp = sp.load_npz(repo+"data/P_events_full.npz")
full_P = tmp.todense().T

print(full_P.shape)

target_C = np.ones((4001))
target_P = np.zeros((4001))

targets = np.zeros(8002)
targets[0:4001] = target_C
targets[4001:] = target_P

data = np.zeros((8002, 8000))
data[0:4001] = full_C
data[4001:] = full_P

X_train, X_test, y_train, y_test = train_test_split(
        data, targets, test_size=0.33, random_state=42, shuffle=True)

lr_model = MLPClassifier(hidden_layer_sizes=(15,))
lr_model.fit(X_train, y_train)

decision_function = lr_model.predict_proba(X_test)[:,1]
fpr, tpr,  _ = roc_curve(y_test, decision_function)
auc = roc_auc_score(y_test, decision_function)
acc = accuracy_score(y_test, lr_model.predict(X_test))

plt.plot(fpr, tpr, label="AUC={:.2f}  ACC={:.2f}".format(auc, acc))
plt.legend()
plt.show()
