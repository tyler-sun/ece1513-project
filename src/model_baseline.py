import numpy as np
from sklearn.linear_model import LogisticRegression


def prepare_features(X):
    mean = np.mean(X, axis=3)
    std = np.std(X, axis=3)

    feat = np.concatenate([mean, std], axis=2)
    feat = feat.reshape(feat.shape[0], -1)

    return feat.astype(np.float32)


def get_model(seed=42):
    return LogisticRegression(
        solver="saga",
        penalty="l2",
        C=0.5,
        max_iter=5,
        warm_start=True,
        random_state=seed,
        n_jobs=-1,
    )