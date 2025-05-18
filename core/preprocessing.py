import pandas as pd

def preprocess(X):
    X = X.copy()
    X = X.dropna()
    X = pd.get_dummies(X, drop_first=True)
    return X
