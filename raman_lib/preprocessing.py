from types import NoneType
from itertools import accumulate

import numpy as np
import pandas as pd
from pybaselines.misc import beads
from pybaselines.morphological import mormol, rolling_ball
from pybaselines.whittaker import arpls, asls
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import normalize
from scipy.signal import savgol_filter, find_peaks


class BaselineCorrector(BaseEstimator, TransformerMixin):
    def __init__(self, method="asls"):
        self.method = method

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        if type(X) != np.ndarray:
            X = np.asarray(X)

        bl = np.zeros_like(X)

        if self.method == "asls":
            for i, row in enumerate(X):
                bl[i] = asls(row)[0]

        elif self.method == "arpls":
            for i, row in enumerate(X):
                bl[i] = arpls(row)[0]

        elif self.method == "mormol":
            for i, row in enumerate(X):
                bl[i] = mormol(row)[0]

        elif self.method == "rolling ball":
            for i, row in enumerate(X):
                bl[i] = rolling_ball(row)[0]

        elif self.method == "beads":
            for i, row in enumerate(X):
                bl[i] = beads(row)[0]

        else:
            raise ValueError(f"Method {self.method} does not exist.")

        return X - bl


class RangeLimiter(BaseEstimator, TransformerMixin):
    def __init__(self, lim=(None, None), reference=None):
        self.lim = lim
        self.reference = reference
        self.lim_ = None

    def fit(self, X, y=None):
        self._validate_params(X)

        if self.reference is not None:
            self.lim_ = np.array(
                [(np.where(self.reference >= l0)[0][0],
                  np.where(self.reference <= l1)[0][-1] + 1)
                 for l0, l1 in zip(self.lim[::2], self.lim[1::2])]
            ).flatten()
        else:
            self.lim_ = np.array(
                [(l0, l1) for l0, l1 in zip(self.lim[::2], self.lim[1::2])]
            ).flatten()
        return self

    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            return pd.concat([X.iloc[:, l0:l1] for l0, l1 in zip(self.lim_[::2], self.lim_[1::2])],
                             axis=1)
        else:
            result = np.concatenate([X[:, l0:l1] for l0, l1 in zip(self.lim_[::2], self.lim_[1::2])],
                                    axis=1)
        return result

    def _replace_nones(self, X):
        if self.lim[0] is None:
            if self.reference is None:
                self.lim[0] = 0
            else:
                self.lim[0] = self.reference[0]

        if self.lim[-1] is None:
            if self.reference is None:
                self.lim[-1] = X.shape[1]
            else:
                self.lim[-1] = self.reference[-1]

    def _validate_params(self, X):
        if self.reference is not None:
            if np.any(self.reference[:-1] > self.reference[1:]):
                raise ValueError("Reference array is not sorted.")
            self.reference = np.asarray(self.reference)

        if self.lim is None:
            self.lim = np.array((None, None), dtype=int)
        else:
            self.lim = np.asarray(self.lim)

        if len(self.lim) % 2 != 0:
            raise ValueError("Odd number of values for limits.")
        if not all([isinstance(val, (int, float, NoneType)) for val in self.lim]):
            raise TypeError("Non-numeric values in limits.")

        if len(self.lim) > 2 and any(val is None for val in self.lim[1:-1]):
            raise ValueError("Only the first and last limit can be None.")

        self._replace_nones(X)

        if np.any([
            self.reference is None and (
                self.lim[0] < 0 or self.lim[1] > X.shape[1]),
            self.reference is not None and (
                self.lim[0] < self.reference[0] or self.lim[1] > self.reference[-1])]):
            raise IndexError(
                "Index out of range. Please check the provided indices.")


class SavGolFilter(BaseEstimator, TransformerMixin):
    """Class to smooth spectral data using a Savitzky-Golay Filter."""

    def __init__(self, window=15, poly=3, limits=None):
        """Initialize window size and polynomial order of the Savitzky-Golay Filter"""
        self.window = window
        self.poly = poly
        self.limits = limits

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.limits is None or len(self.limits) == 2:
            X_smooth = savgol_filter(
                X, window_length=self.window, polyorder=self.poly)
            X_smooth = (X_smooth.T - X_smooth.min(axis=1)).T
        else:
            breaks = np.cumsum([l1 - l0 for l0, l1 in zip(self.limits[::2],
                                                          self.limits[1::2])])[:-1]
            parts = np.split(X, breaks, axis=1)
            X_smooth = []
            for part in parts:
                X_smooth.append(savgol_filter(part,
                                              window_length=self.window,
                                              polyorder=self.poly))
                X_smooth[-1] = (X_smooth[-1].T - X_smooth[-1].min(axis=1)).T
            X_smooth = np.concatenate(X_smooth, axis=1)

        return X_smooth


class Normalizer(BaseEstimator, TransformerMixin):
    def __init__(self, norm="l2", limits=None, copy=True):
        self.norm = norm
        self.limits = limits
        self.copy = copy

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.limits is None:
            return normalize(X, norm=self.norm, copy=self.copy)
        else:
            breaks = np.cumsum([l1 - l0 for l0, l1 in zip(self.limits[::2],
                                                          self.limits[1::2])])[:-1]
            # Split the array, normalize each part and concatenate
            parts = np.split(X, breaks, axis=1)
            X_norm = []
            for part in parts:
                X_norm.append(normalize(part, norm=self.norm, copy=self.copy))
            return np.concatenate(X_norm, axis=1)


class PeakPicker(BaseEstimator, TransformerMixin):
    def __init__(self, min_dist=None):
        self.min_dist = min_dist

    def fit(self, X, y=None):
        X_mean = X.mean(axis=0)
        self.peak_indices = find_peaks(X_mean, distance=self.min_dist)[0]
        self.peaks_ = np.zeros((len(self.peak_indices), X.shape[1]), dtype=bool)
        for i, j in enumerate(self.peak_indices):
            self.peaks_[i, j] = True
        return self

    def transform(self, X, y=None):
        return X[:, self.peak_indices]
