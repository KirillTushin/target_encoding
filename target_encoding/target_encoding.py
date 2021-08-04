"""Module for target encoding."""

from typing import Union, Tuple, List

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import check_cv, BaseCrossValidator

from target_encoding.utils import TargetStatistic, cv_splitter

__version__ = '1.1.0'


class BaseTargetEncoder:
    """Class for encoding one data by another data."""

    def __init__(self, alpha: float = 0.0, max_bins: int = 30):
        """

        Args:
            alpha (float): smoothing parameter for generalization.
            max_bins (int): maximum number of unique values in a feature.
        """

        self.statistic = TargetStatistic(
            alpha=alpha,
            max_bins=max_bins,
        )

    def fit(
            self,
            x_array: np.ndarray,
            target_array: np.ndarray,
    ) -> None:
        """Fit statistics for target encoding.

        Args:
            x_array (np.ndarray): data to transform by target encoding.
            target_array (np.ndarray): targets for encoding "x_array".

        Returns:
            None
        """
        self.statistic.fit(x_array, target_array)

    def transform(self, x_array: np.ndarray) -> np.ndarray:
        """Transform data by fitted statistics.

        Args:
            x_array (np.ndarray): data to transform by target encoding.

        Returns:
            np.ndarray: Transformed data.
        """

        if not self.statistic.is_fitted:
            raise UserWarning("Model must be trained before transform.")

        if self.statistic.bins is not None:
            x_array = np.digitize(x_array, self.statistic.bins)

        transformed_x_array = []
        for value in x_array:
            transformed_x_array.append(
                self.statistic.map_cat.get(
                    value,
                    self.statistic.global_mean,
                )
            )
        return np.array(transformed_x_array, float)

    def __str__(self):
        return f'{self.__class__.__name__}'


class TargetEncoder(BaseEstimator):
    """Target encoding transformer and base class
        for Classifications and Regressions problems.
    """

    def __init__(
            self,
            alpha: float = 10,
            max_bins: int = 30,
            split: Tuple[Union[int, BaseCrossValidator]] = (3, 3),
    ):
        """

        Args:
            alpha (float): smoothing parameter for generalization.
            max_bins (int): maximum number of unique values in a feature.
            split (tuple[Union[int, BaseCrossValidator]): tuple of int or
                cross-validator classes.

                If split len is 0, then algorithm
                will encode features without cross-validation.
                This situation features will over-fit on target.

                If split len is 1, algorithm will encode features by using
                cross-validation on folds.
                In this situation you will not over-fit on tests,
                but when you will validate, your score may over-fit.

                If split len is 2, algorithm will separate data on first folds,
                afterwords will encode features by using cross-validation
                on second folds. This situation is the best way to
                avoid over-fit, but algorithm will use small data for encode.
        """

        self.alpha = alpha
        self.max_bins = max_bins
        self.split = tuple(check_cv(x_split) for x_split in split)
        self._encodings = []  # type: List[BaseTargetEncoder]

    def _encode_one_feature(
            self,
            x_array: np.ndarray,
            target_array: np.ndarray,
    ) -> BaseTargetEncoder:
        """Fit statistic for one feature in dataset.

        Args:
            x_array (np.ndarray): data to transform by target encoding.
            target_array (np.ndarray): targets for encoding "x_array".

        Returns:
            BaseTargetEncoder: Trained encoder is returned.
        """

        enc = BaseTargetEncoder(self.alpha, self.max_bins)
        enc.fit(x_array, target_array)
        return enc

    def fit(
            self,
            dataset: np.ndarray,
            target_array: np.ndarray,
    ) -> None:
        """Fit statistics for each feature in dataset.

        Args:
            dataset (np.ndarray): Dataset for encoding,
                has to have (n_rows, n_columns) shape.
            target_array (np.ndarray): targets for encoding "dataset".

        Returns:
            None
        """

        self._encodings = []
        for i in range(dataset.shape[1]):
            self._encodings.append(
                self._encode_one_feature(dataset[:, i], target_array)
            )

    def transform_train(
            self,
            dataset: np.ndarray,
            target_array: np.ndarray,
    ) -> np.ndarray:
        """Transform train data with split and fit statistics.

        Args:
            dataset (np.ndarray): Dataset for encoding,
                has to have (n_rows, n_columns) shape.
            target_array (np.ndarray): targets for encoding "dataset".

        Returns:
            np.ndarray: Transformed train data.
        """

        new_dataset = np.zeros(dataset.shape)
        enc = BaseTargetEncoder(self.alpha, self.max_bins)

        indexes = np.array(range(dataset.shape[0]))

        for (tr_index, val_index) in cv_splitter(
                indexes, target_array, self.split,
        ):
            for i in range(dataset.shape[1]):
                x_column = dataset[:, i]

                enc.fit(x_column[tr_index], target_array[tr_index])
                new_dataset[val_index, i] = enc.transform(x_column[val_index])

        self.fit(dataset, target_array)
        return new_dataset

    def transform_test(self, dataset: np.ndarray) -> np.ndarray:
        """Transform test data by fitted statistics.

        Args:
            dataset (np.ndarray): Dataset for encoding,
                has to have (n_rows, n_columns) shape.

        Returns:
            np.ndarray: Transformed test data.
        """

        if dataset.shape[1] != len(self._encodings):
            raise ValueError(
                f'Number of columns in train was {len(self._encodings)} \
                and Number of columns in tests {dataset.shape[1]}')

        new_dataset = np.zeros(dataset.shape)
        for i in range(dataset.shape[1]):
            enc = self._encodings[i]
            new_dataset[:, i] = enc.transform(dataset[:, i])
        return new_dataset

    def __str__(self):
        return f'{self.__class__.__name__}'


class TargetEncoderRegressor(TargetEncoder):
    """Class based on TargetEncoder for Regression problems."""

    def __init__(
            self,
            alpha: float = 10,
            max_bins: int = 30,
            used_features: int = 10,
    ):
        """

        Args:
            alpha (float): smoothing parameter for generalization.
            max_bins (int): maximum number of unique values in a feature.
            used_features (int): Number of used features for prediction.
                Value has to be between 1 and number of features.
        """

        super().__init__(alpha, max_bins)
        self.used_features = used_features

    def predict(self, dataset: np.ndarray) -> np.ndarray:
        """Prediction for each object by using target encoding.

        Args:
            dataset (np.ndarray): Dataset for encoding,
                has to have (n_rows, n_columns) shape.

        Returns:
            np.ndarray: Target encoded objects.
        """

        new_x = self.transform_test(dataset)
        use_features = np.argsort(new_x.std(axis=0))[-self.used_features:]

        mean = np.mean(new_x[:, use_features], axis=1)
        return mean


class TargetEncoderClassifier(TargetEncoderRegressor):
    """Class based on TargetEncoder for Classification problems."""

    def predict_proba(self, dataset: np.ndarray) -> np.ndarray:
        """Prediction for each object by using target encoding.

        Args:
            dataset (np.ndarray): Dataset for encoding,
                has to have (n_rows, n_columns) shape.

        Returns:
            np.ndarray: Target encoded objects.
        """
        pred = super().predict(dataset)

        return np.array([1 - pred, pred]).T

    def predict(self, dataset: np.ndarray) -> np.ndarray:
        """Class prediction for each object by using target encoding.

        Args:
            dataset (np.ndarray): Dataset for encoding,
                has to have (n_rows, n_columns) shape.

        Returns:
            np.ndarray: Classes by using argmax from predict_proba method.
        """

        return np.argmax(self.predict_proba(dataset), axis=1)
