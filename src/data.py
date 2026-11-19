#standardizes with TRAIN stats only and adds the bias column of 1s at the end
import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split



def load_task(name, test_size = 0.25, seed = 0):

    if name == "digits01":
        d = load_digits()
        #only keep the 0s and 1s
        X = d.data[d.target <= 1]
        y = d.target[d.target <= 1]

    elif name == "breast_cancer":
        X, y = load_breast_cancer(return_X_y=True)

    else:
        raise ValueError("unknown task " + name)


    #75/25 split, stratified so both sides keep the same class balance
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

    # standardize using the training data only
    mean = Xtr.mean(axis=0)
    std = Xtr.std(axis=0)
    std[std == 0] = 1.0   #some digits pixels are always 0, this keeps them from dividing by 0

    def prep(A):
        #scale then stick a column of 1s on the end for the bias
        return np.hstack([ (A - mean) / std, np.ones((len(A), 1)) ])


    #info the spec wants us to report
    info = dict(task=name, N_train=len(ytr), N_test=len(yte), D=X.shape[1],
                train_counts=np.bincount(ytr).tolist(), test_counts=np.bincount(yte).tolist())

    return prep(Xtr), prep(Xte), ytr, yte, info
