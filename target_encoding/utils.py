"""Utils for target encoding module"""

from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Any, Union, Tuple, Dict, List

import numpy as np
from sklearn.model_selection import check_cv, BaseCrossValidator


@dataclass
class TargetStatistic:
    """Class for keeping track of an item in inventory."""
    alpha: float = 0
    max_bins: int = 30
    map_cat: Dict[Any, float] = field(
        default_factory=defaultdict(lambda: 0.5)
    )
    global_mean: float = 0.5
    bins: list = field(default_factory=list())
    is_fitted: bool = False

    def __init__(self, alpha: float = 0.0, max_bins: int = 30):
        """

        Args:
            alpha (float): smoothing parameter for generalization.
            max_bins (int): maximum number of unique values in a feature.
        """
        self.alpha = alpha
        self.max_bins = max_bins

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
        bins = None
        map_cat = defaultdict(lambda: 0.5)
        global_mean = np.mean(target_array)

        len_unique = len(np.unique(x_array))

        if len_unique >= self.max_bins:
            bins = np.histogram_bin_edges(x_array, self.max_bins)
            x_array = np.digitize(x_array, bins)

        for cat in Counter(x_array):
            index_for_target = np.where(x_array == cat)[0]
            nrows = len(index_for_target)
            mean_cat = np.mean(target_array[index_for_target])
            map_cat[cat] = (mean_cat * nrows + global_mean * self.alpha) \
                           / (nrows + self.alpha)

        self.bins = bins
        self.map_cat = map_cat
        self.global_mean = global_mean
        self.is_fitted = True


def cv_splitter(
        indexes: np.ndarray,
        target_array: np.ndarray,
        split: Tuple[Union[int, BaseCrossValidator]] = (3, 3),
) -> List[Tuple[List[int], List[int]]]:
    """Function to create train and test indexes to fit target encoding.
        Args:
            indexes (npt.ArrayLike): Indexes of dataset for encoding.
            target_array (npt.ArrayLike): targets for encoding "x_array".

            split (tuple[Union[int, BaseCrossValidator]): tuple of int or
                cross-validator classes.

                If split len is 0, then algorithm
                will encode features without cross-validation.
                This situation features will over-fit on target.

                If split len is 1, algorithm will encode features by using
                cross-validation on folds.
                In this situation you will not overfit on tests,
                but when you will validate, your score may over-fit.

                If split len is 2, algorithm will separate data on first folds,
                afterwords will encode features by using cross-validation
                on second folds. This situation is the best way to
                avoid over-fit, but algorithm will use small data for encode.

        Returns:
            List[Tuple[List[int], List[int]]]: Train and Test indexes to fit
                target encoding.
    """

    split = tuple(check_cv(x_split) for x_split in split)
    len_of_split = len(split)

    if len_of_split == 0:
        return [(indexes, indexes)]

    result = []
    for cv_split in split:
        for tr_index, val_index in cv_split.split(indexes, target_array):
            if len_of_split > 1:
                result.extend(
                    cv_splitter(
                        indexes=indexes[val_index],
                        target_array=target_array[val_index],
                        split=tuple(x_split for x_split in split[1:])
                    )
                )
            return [(tr_index, val_index)]
