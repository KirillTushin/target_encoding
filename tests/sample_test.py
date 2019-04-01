from target_encoding import TargetEncoder, TargetEncoderClassifier, TargetEncoderRegressor
import numpy as np
import pandas as pd

from itertools import product

FILLNA = -(10**7 + 1)


def get_out(X_pd, y, alpha):
    X_pd = X_pd.fillna(FILLNA)
    X_pd['y'] = y
    global_mean = X_pd['y'].mean()

    for col in X_pd.columns:
        dict_res = {}

        for i in np.unique(X_pd[col]):
            n_el = X_pd.groupby(col)['y'].count()[i]
            dict_res[i] = (X_pd.groupby(col)['y'].sum()[i] + global_mean * alpha) / (alpha + n_el)
        X_pd[col] = X_pd[col].map(dict_res)
    X_pd = X_pd.drop('y', axis=1)
    return X_pd.values


def generate_data_alpha(alpha = 0):
    X_1_ls = [
        [0],
        [1.1],
        [0],
        [np.nan],
        [np.nan],
        [0],
        [1.1],
        [np.nan],
        [0],
        [2],
    ]
    X_1_np = np.array(X_1_ls).reshape(-1, 1)
    X_1_pd = pd.DataFrame(X_1_ls)

    X_2_ls = [
        [3, 100],
        [109, 80.5],
        [3, np.nan],
        [np.nan, 37],
        [np.nan, 37],
        [3, 80.5],
        [109, 37],
        [np.nan, 80.5],
        [2.4, 37],
        [2.4, np.nan],
    ]
    X_2_np = np.array(X_2_ls)
    X_2_pd = pd.DataFrame(X_2_ls)

    y_ls = [1, 1, 0, 1, 0, 1, 1, 0, 1, 0]
    y_np = np.array(y_ls)
    y_pd = pd.DataFrame(y_ls)

    out_1 = get_out(X_1_pd, y_ls, alpha)
    out_2 = get_out(X_2_pd, y_ls, alpha)

    return X_1_ls, X_1_np, X_1_pd, X_2_ls, X_2_np, X_2_np, X_2_pd, y_ls, y_np, y_pd, out_1, out_2


def generate_data_init():
    X = [
        [3, 100],
        [109, 80.5],
        [3, np.nan],
        [np.nan, 37],
        [np.nan, 37],
        [3, 80.5],
        [109, 37],
        [np.nan, 80.5],
        [2.4, 37],
        [2.4, np.nan],
    ]
    X = np.concatenate([X for _ in range(100)], axis=1)
    X = np.concatenate([X for _ in range(100)], axis=0)

    y = [1, 1, 0, 1, 0, 1, 1, 0, 1, 0]
    y = np.concatenate([y for _ in range(100)], axis=0)

    return X, y


def test_input_transform():

    for alpha in np.arange(0, 1000, 10):
        X_1_ls, X_1_np, X_1_pd, X_2_ls, X_2_np, X_2_np, X_2_pd, y_ls, y_np, y_pd, out_1, out_2 = generate_data_alpha(alpha)

        enc = TargetEncoder(alpha=alpha, max_unique=30, split=[])

        for y in [y_ls, y_np, y_pd]:
            for X in [X_1_ls, X_1_np, X_1_pd]:
                assert (enc.transform_train(X, y) == out_1).all()

        for y in [y_ls, y_np, y_pd]:
            for X in [X_2_ls, X_2_np, X_2_pd]:
                assert (enc.transform_train(X, y) == out_2).all()


def test_correct_init_encoder():

    X, y = generate_data_init()

    for alpha in np.arange(0, 100, 10):
        enc = TargetEncoder(alpha=alpha)
        enc.transform_train(X, y)
        enc.transform_test(X)

    for max_unique in np.arange(2, 100, 10):
        enc = TargetEncoder(max_unique=max_unique)
        enc.transform_train(X, y)
        enc.transform_test(X)

    for split_1, split_2 in product(range(1, 6), range(1, 6)):

        split = []
        if split_1 == 1:
            continue
        else:
            split.append(split_1)
        if split_2 != 1:
            split.append(split_2)

        enc = TargetEncoder(split=split)
        enc.transform_train(X, y)
        enc.transform_test(X)


def test_correct_init_classifier():
    X, y = generate_data_init()

    for alpha in np.arange(0, 100, 10):
        enc = TargetEncoderClassifier(alpha=alpha)
        enc.fit(X, y)
        enc.predict(X)
        enc.predict_proba(X)

    for max_unique in np.arange(2, 100, 10):
        enc = TargetEncoderClassifier(max_unique=max_unique)
        enc.fit(X, y)
        enc.predict(X)
        enc.predict_proba(X)

    for used_features in np.arange(1, 200, 10):
        enc = TargetEncoderClassifier(used_features=used_features)
        enc.fit(X, y)
        enc.predict(X)
        enc.predict_proba(X)


def test_correct_init_regressor():
    X, y = generate_data_init()

    for alpha in np.arange(0, 100, 10):
        enc = TargetEncoderRegressor(alpha=alpha)
        enc.fit(X, y)
        enc.predict(X)

    for max_unique in np.arange(2, 100, 10):
        enc = TargetEncoderRegressor(max_unique=max_unique)
        enc.fit(X, y)
        enc.predict(X)

    for used_features in np.arange(1, 200, 10):
        enc = TargetEncoderRegressor(used_features=used_features)
        enc.fit(X, y)
        enc.predict(X)
