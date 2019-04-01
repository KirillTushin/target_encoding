import numpy as np
from sklearn.base import BaseEstimator
from target_encoding.utils import validate_input, _generate_map_cat, _transform_array, _bin_x

__version__ = '0.5.0'


class _BaseTargetEncoder:
    """
    Class for encoding one data by another data.
    """

    def __init__(self, alpha=10, max_unique=30):
        """

        :param alpha: float or int, smoothing for generalization.
        :param max_unique: int, maximum number of unique values in a feature.
        """
        val = validate_input(alpha=alpha, max_unique=max_unique)
        self.alpha = val['alpha']
        self.max_unique = val['max_unique']
        self._map_cat = {}
        self._global_mean = None
        self._bins = None
        self._is_fitted = False

    def fit(self, x, y):
        """

        :param x: array like data of objects.
        :param y: array like data of targets, targets have to be int or float type.
        :return: None.
        """

        val = validate_input(x=x, y=y)
        x = val['x']
        y = val['y']

        self._bins, x = _bin_x(x, self.max_unique)

        self._map_cat = _generate_map_cat(x, y, self.alpha)
        self._global_mean = np.mean(y)
        self._is_fitted = True

    def transform(self, x):
        """

        :param x: array like data of objects.
        :return: array where old values from x replace on new values from self._map_cat.
        """

        val = validate_input(x=x)
        x = val['x']

        if self._is_fitted is False:
            raise UserWarning("you have to fit model before transform")

        if self._bins is not None:
            x = np.digitize(x, self._bins)

        return _transform_array(self._map_cat, x, self._global_mean)

    def __str__(self):
        return f'{self.__class__.__name__}'


class TargetEncoder(BaseEstimator):
    """
    Target encoding transformer and base class for Classifications and Regressions problems.
    """
    def __init__(self, alpha=10, max_unique=30, split=[3, 3]):
        """

        :param alpha: float or int, smoothing for generalization.
        :param max_unique: int, maximum number of unique values in a feature.
        :param split: list of int or cross-validator class,
                if split is [], then algorithm will encode features without cross-validation
                This situation features will overfit on target

                if split len is 1 for example [5], algorithm will encode features by using cross-validation on 5 folds
                This situation you will not overfit on tests, but when you will validate, your score will overfit

                if split len is 2 for example [5, 3], algorithm will separate data on 5 folds, afterwords
                will encode features by using cross-validation on 3 folds
                This situation is the best way to avoid overfit, but algorithm will use small data for encode.
        """

        val = validate_input(alpha=alpha, max_unique=max_unique, split=split)
        self.alpha = val['alpha']
        self.max_unique = val['max_unique']
        self.split = val['split']
        self._encodings = []
        self._is_fitted = False

    def _encode_one_feature(self, x, y):
        """

        :param x: array like data of objects.
        :param y: array like data of targets, targets have to be int or float type.
        :return: _BaseTargetEncoder.
        """
        val = validate_input(x=x, y=y)
        x = val['x']
        y = val['y']

        enc = _BaseTargetEncoder(self.alpha, self.max_unique)
        enc.fit(x, y)
        return enc

    def _fit(self, X, y):
        """

        :param X: array like data for encoding, X has to have (n_rows, n_columns) shape.
        :param y: array like data of targets, targets have to be int or float type.
        :return: None
        """

        val = validate_input(X=X, y=y)
        X = val['X']
        y = val['y']

        self._encodings = []
        for i in range(X.shape[1]):
            self._encodings.append(self._encode_one_feature(X[:, i], y))

        self._is_fitted = True

    def transform_train(self, X, y):
        """

        :param X: array like data for encoding, X has to have (n_rows, n_columns) shape.
        :param y: array like data of targets, targets have to be int or float type.
        :return: array where old values from X replace on new values from self._map_cat.
        """

        val = validate_input(X=X, y=y)
        X = val['X']
        y = val['y']

        new_X = np.zeros(X.shape)

        for i in range(X.shape[1]):
            x_col = X[:, i]

            if len(self.split) == 0:
                enc = _BaseTargetEncoder(self.alpha, self.max_unique)
                enc.fit(x_col, y)
                new_X[:, i] = enc.transform(x_col)

            if len(self.split) == 1:
                cv = self.split[0]
                for tr_index, val_index in cv.split(x_col, y):
                    enc = _BaseTargetEncoder(self.alpha, self.max_unique)
                    enc.fit(x_col[tr_index], y[tr_index])
                    new_X[val_index, i] = enc.transform(x_col[val_index])

            if len(self.split) == 2:
                cv_0 = self.split[0]
                cv_1 = self.split[1]
                for tr_index_1, val_index_1 in cv_0.split(x_col, y):
                    for tr_index_2, val_index_2 in cv_1.split(x_col[val_index_1], y[val_index_1]):
                        enc = _BaseTargetEncoder(self.alpha, self.max_unique)

                        tr_index = val_index_1[tr_index_2]
                        val_index = val_index_1[val_index_2]

                        enc.fit(x_col[tr_index], y[tr_index])
                        new_X[val_index, i] = enc.transform(x_col[val_index])

        self._fit(X, y)
        return new_X

    def transform_test(self, X):
        """

        :param X: array like data to be encoded, X has to have (n_rows, n_columns) shape.
        :return: array where old values from X replace on new values from self._map_cat.
        """

        val = validate_input(X=X)
        X = val['X']

        if self._is_fitted is False:
            raise UserWarning("you have to fit model before transform")

        if X.shape[1] != len(self._encodings):
            raise ValueError(
                f'count of columns in train was {len(self._encodings)} and count of columns in tests {X.shape[1]}')

        new_X = np.zeros(X.shape)
        for i in range(X.shape[1]):
            enc = self._encodings[i]
            new_X[:, i] = enc.transform(X[:, i])
        return new_X

    def __str__(self):
        return f'{self.__class__.__name__}'


class TargetEncoderRegressor(TargetEncoder):

    def __init__(self, alpha=10, max_unique=30, used_features=10):
        """

        :param alpha: float or int, smoothing for generalization.
        :param max_unique: int, maximum number of unique values in a feature.
        :param used_features: int, This is a number of used features for prediction
                minimum value has to be 1 and  if value more than number of features, will be used all features.

        """

        val = validate_input(alpha=alpha, max_unique=max_unique, used_features=used_features)
        val['alpha'] = alpha
        val['max_unique'] = max_unique
        val['used_features'] = used_features

        super().__init__(alpha, max_unique)
        self.used_features = used_features

    def fit(self, X, y):
        """

        :param X: array like data for encoding, X has to have (n_rows, n_columns) shape.
        :param y: array like data of targets, targets have to be int or float type.
        :return: None.
        """
        val = validate_input(X=X, y=y)
        X = val['X']
        y = val['y']

        super()._fit(X, y)
        self._is_fitted = True

    def decision_function(self, X):
        """

        :param X: array like data for prediction.
        :return: mean value of target encoding for objects.
        """
        val = validate_input(X=X)
        X = val['X']

        new_x = self.transform_test(X)
        use_features = np.argsort(new_x.std(axis=0))[-self.used_features:]

        mean = np.mean(new_x[:, use_features], axis=1)
        return mean

    def predict(self, X):
        return self.decision_function(X)


class TargetEncoderClassifier(TargetEncoderRegressor):
    """
    Class based on TargetEncoder for classification problems.
    """
    def predict_proba(self, X):
        """

        :param X: array like data for prediction.
        :return: probability of classes.
        """

        val = validate_input(X=X)
        X = val['X']
        pred = super().decision_function(X)

        return np.array([1 - pred, pred]).T

    def predict(self, X):
        """

        :param X:array like data for prediction.
        :return: predicted classes.
        """

        val = validate_input(X=X)
        X = val['X']

        return np.argmax(self.predict_proba(X), axis=1)
