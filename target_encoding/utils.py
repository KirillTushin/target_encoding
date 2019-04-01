import collections
from sklearn.utils import check_array
from sklearn.model_selection import check_cv
import numpy as np

FILLNA = -(10**7 + 1)


def validate_input(X=None, x=None, y=None, alpha=None, map_cat=None, global_mean=None,
                    max_unique=None, split=None, used_features=None):
    """

    :param X: array like data for encoding, X has to have (n_rows, n_columns) shape.
    :param x: array like data of objects.
    :param y: array like data of targets, targets have to be int or float type.
    :param alpha: float or int smoothing for generalization.
    :param map_cat: dict, which contain mean value of y for each unique value in x.
    :param global_mean: float or int, mean value of y for all data.
    :param max_unique: int, maximum number of unique values in a feature.
    :param used_features: int, This is a number of used features for prediction
                minimum value has to be 1 and  if value more than number of features, will be used all features.
    :param split: list of int or cross-validator class,
                if split is [], then algorithm will encode features without cross-validation
                This situation features will overfit on target

                if split len is 1 for example [5], algorithm will encode features by using cross-validation on 5 folds
                This situation you will not overfit on tests, but when you will validate, your score will overfit

                if split len is 2 for example [5, 3], algorithm will separate data on 5 folds, afterwords
                will encode features by using cross-validation on 3 folds
                This situation is the best way to avoid overfit, but algorithm will use small data for encode.
    :return: dict of validated data
    """
    output = {}

    if X is not None:
        try:
            X = check_array(X, ensure_2d=True, force_all_finite=False)
            X = np.array(X, np.float64)
            mask = np.isnan(X)
            X[mask] = FILLNA
        except TypeError:
            raise TypeError("X has to be array like data")

        except ValueError:
            raise TypeError("X has to be 2d-array like data of float or int")

        output['X'] = X

    if x is not None:
        try:
            x = check_array(x, ensure_2d=False, force_all_finite=False)
            x = np.array(x, np.float64)
            mask = np.isnan(x)
            x[mask] = FILLNA
        except TypeError:
            raise TypeError("x has to be array like data")

        except ValueError:
            raise TypeError("X has to be array like data of float or int")
        output['x'] = x

    if y is not None:
        try:
            y = check_array(y, ensure_2d=False)
        except TypeError:
            raise TypeError("y has to be array like data without nan")
        output['y'] = y

    if alpha is not None:
        if not isinstance(alpha, (int, float,
                                  np.int64, np.float64,
                                  np.int32, np.float32,
                                  np.int16, np.float16,
                                  np.int8, np.int0)):
            raise ValueError("alpha has to be float or int")
        output['alpha'] = alpha

    if map_cat is not None:
        if not isinstance(map_cat, dict):
            raise ValueError("map_cat has to be dict")
        output['map_cat'] = map_cat

    if global_mean is not None:
        if not isinstance(global_mean, (int, float,
                                  np.int64, np.float64,
                                  np.int32, np.float32,
                                  np.int16, np.float16,
                                  np.int8, np.int0)):
            raise ValueError("global_mean has to be float or int")
        output['global_mean'] = global_mean

    if max_unique is not None:
        if not isinstance(max_unique, (int,  np.int64,
                                  np.int32, np.int16,
                                  np.int8, np.int0)):
            raise ValueError("max_unique has to be int")
        output['max_unique'] = max_unique

    if split is not None:
        if type(split) is not list:
            raise TypeError("split has to be list of cv")
        if len(split) > 2:
            raise TypeError("split type has to have len < 3")

        out_split_type = []
        for x in split:
            try:
                out_split_type.append(check_cv(x))
            except TypeError:
                raise TypeError("cv in split has to be int or cross-validator class")

        output['split'] = out_split_type

    if used_features is not None:
        if not isinstance(used_features, (int,  np.int64,
                                  np.int32, np.int16,
                                  np.int8, np.int0)):
            raise ValueError("max_unique has to be int")
        if used_features < 1:
            raise ValueError("minimal value fo used_features has to be 1")
        output['used_features'] = used_features

    return output


def _generate_map_cat(x, y, alpha=10):
    """

    :param x: array like data of objects.
    :param y: array like data of targets, targets have to be int or float type.
    :param alpha: float or int smoothing for generalization.
    :return: dict, which contain mean value of y for each unique value in x.
    """

    val = validate_input(x=x, y=y, alpha=alpha)
    x = val['x']
    y = val['y']
    alpha = val['alpha']

    global_mean = np.mean(y)
    map_cat = {}

    for cat in collections.Counter(x):
        index_for_target = np.where(x == cat)[0]
        nrows = len(index_for_target)
        mean_cat = np.mean(y[index_for_target])
        map_cat[cat] = (mean_cat * nrows + global_mean * alpha) / (nrows + alpha)
    return map_cat


def _transform_array(map_cat, x, global_mean):
    """

    :param map_cat: dict, which contain mean value of y for each unique value in x.
    :param x: array like data of objects.
    :param global_mean: float or int, mean value of y for all data.
    :return: array where old values from x replace on new values from map_cat.
    """

    val = validate_input(map_cat=map_cat, x=x, global_mean=global_mean)
    map_cat = val['map_cat']
    x = val['x']
    global_mean = val['global_mean']

    new_array = []
    for x_ in x:
        if x_ in map_cat:
            new_array.append(map_cat[x_])
        else:
            new_array.append(global_mean)
    return np.array(new_array, float)


def _bin_x(x, max_unique):
    """

    :param x: array like data of objects.
    :param max_unique: int, maximum number of unique values in a feature.
    :return: bins and binned x.
    """
    val = validate_input(x=x, max_unique=max_unique)
    x = val['x']
    max_unique = val['max_unique']

    bins = None
    len_unique = len(np.unique(x))

    if len_unique >= max_unique:
        bins = np.histogram_bin_edges(x, bins=max_unique)
        x = np.digitize(x, bins)
    return bins, x
