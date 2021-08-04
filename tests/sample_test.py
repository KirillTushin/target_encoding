"""Test target encoding module"""

from itertools import product

import numpy as np
import pandas as pd

from target_encoding import TargetEncoder
from target_encoding import TargetEncoderClassifier, TargetEncoderRegressor

TARGET = np.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
FEATURE = np.array([0, 1.1, 0, 7, 7, 0, 1.1, 7, 0, 2])
DATASET = np.array([
    [3, 100],
    [109, 80.5],
    [3, 8],
    [10, 37],
    [10, 37],
    [3, 80.5],
    [109, 37],
    [10, 80.5],
    [2.4, 37],
    [2.4, 8],
])
HUGE_DATASET = np.concatenate([DATASET for _ in range(100)], axis=1)
HUGE_DATASET = np.concatenate([HUGE_DATASET for _ in range(100)], axis=0)
HUGE_TARGET = np.concatenate([TARGET for _ in range(100)], axis=0)


def get_out(dataset, target, alpha):
    """Get target for different alpha"""
    dataset = pd.DataFrame(dataset)
    dataset['target'] = target
    global_mean = dataset['target'].mean()

    for col in dataset.columns:
        dict_res = {}
        count_groupby = dataset.groupby(col)['target'].count()
        sum_groupby = dataset.groupby(col)['target'].sum()

        for value in dataset[col].unique():
            n_el = count_groupby[value]
            dict_res[value] = (sum_groupby[value] + global_mean * alpha) \
                              / (alpha + n_el)
        dataset[col] = dataset[col].map(dict_res)
    dataset = dataset.drop('target', axis=1)
    return dataset.values


def test_input_transform():
    """Test result of target encoding"""
    for alpha in np.arange(0, 1000, 10):
        out_feature = get_out(FEATURE, TARGET, alpha)
        out_dataset = get_out(DATASET, TARGET, alpha)

        enc = TargetEncoder(alpha=alpha, max_bins=30, split=())

        result = enc.transform_train(FEATURE.reshape(-1, 1), TARGET)
        assert (result == out_feature).all()

        result = enc.transform_train(DATASET, TARGET)
        assert (result == out_dataset).all()


def test_correct_init():
    """Test call of target encoding module"""
    for alpha in np.arange(0, 31, 10):
        for max_bins in np.arange(1, 100, 30):
            for model in [TargetEncoderClassifier, TargetEncoderRegressor]:
                enc = model(alpha=alpha, max_bins=max_bins)
                enc.transform_train(HUGE_DATASET, HUGE_TARGET)
                enc.transform_test(HUGE_DATASET)

            for split_1, split_2 in product(range(1, 4), range(1, 4)):
                split = []
                if split_1 != 1:
                    split.append(split_1)

                if split_2 != 1:
                    split.append(split_2)

                split = tuple(split)

                enc = TargetEncoder(alpha=alpha, max_bins=max_bins, split=split)
                enc.transform_train(HUGE_DATASET, HUGE_TARGET)
                enc.transform_test(HUGE_DATASET)
