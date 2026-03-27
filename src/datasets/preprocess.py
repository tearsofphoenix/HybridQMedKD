import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE


class FoldPreprocessor:
    """
    Fit-transform inside each training fold to prevent data leakage.
    Supports standard scaling + optional PCA dimensionality reduction.
    """

    def __init__(self, pca_dim=4, balance_method="none", random_state=42):
        self.pca_dim = pca_dim
        self.balance_method = balance_method
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.reducer = PCA(n_components=pca_dim) if pca_dim is not None else None

    def fit_transform(self, X_train, y_train=None):
        Xs = self.scaler.fit_transform(X_train)
        if self.reducer is not None:
            Xs = self.reducer.fit_transform(Xs)
        if self.balance_method == "smote" and y_train is not None:
            sm = SMOTE(random_state=self.random_state)
            Xs, y_train = sm.fit_resample(Xs, y_train)
        return Xs, y_train

    def transform(self, X):
        Xs = self.scaler.transform(X)
        if self.reducer is not None:
            Xs = self.reducer.transform(Xs)
        return Xs
