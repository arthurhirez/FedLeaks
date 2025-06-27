import warnings

import numpy as np
import pandas as pd


def intervals_to_mask(index, intervals):
    """Create boolean mask from given intervals.

    The function creates an boolean array of same size as the given index array.
    If an index value is within a given interval, the corresponding mask value is `True`.

    Args:
        index (ndarray):
            Array containing the index values.
        intervals (list or ndarray):
            List or array of intervals, consisting of start-index and end-index for each interval.

    Returns:
        ndarray:
            Array of boolean values, with one boolean value for each index value (`True` if the
            index value is contained in a given interval, otherwise `False`).
    """
    mask = list()

    if intervals is None or len(intervals) == 0:
        return np.zeros((len(index), ), dtype=bool)

    for value in index:
        for start, end in intervals:
            if start <= value <= end:
                mask.append(True)
                break
        else:
            mask.append(False)

    return np.array(mask)


def rolling_window_sequences(
    X,
    index,
    window_size,
    target_size,
    step_size,
    target_column,
    offset=0,
    drop=None,
    drop_windows=False,
    return_window_index=False,
    return_center_index=False
):
    """
    Create rolling window sequences from time series data.

    Args:
        X (ndarray): Input data (N x D).
        index (ndarray): Timestamps for each row in `X` (length N).
        window_size (int): Number of steps per input window.
        target_size (int): Number of steps in the target sequence.
        step_size (int): Step size between windows.
        target_column (int or list): Column(s) of X to use as target.
        offset (int): Gap between input and target.
        drop (ndarray, float, str, bool, or None): Optional drop mask or value.
        drop_windows (bool): If True, drop windows with invalid values.
        return_window_index (bool): If True, return full list of timestamps for each input window.
        return_center_index (bool): If True, return center timestamp of each window.

    Returns:
        tuple: (X_seq, y_seq, X_index, y_index[, window_index][, center_index])
    """
    out_X = []
    out_y = []
    X_index = []
    y_index = []
    window_index = []
    center_index = []

    target = X[:, target_column] if isinstance(target_column, int) else X[:, target_column]

    if drop_windows:
        if hasattr(drop, '__len__') and not isinstance(drop, str):
            if len(drop) != len(X):
                raise ValueError("Length of `drop` must match X.")
        else:
            drop = np.isnan(X) if isinstance(drop, float) and np.isnan(drop) else X == drop

    start = 0
    max_start = len(X) - window_size - target_size - offset + 1

    while start < max_start:
        end = start + window_size

        if drop_windows:
            drop_window = drop[start:end + target_size]
            if np.any(drop_window):
                start += np.where(drop_window)[0][-1] + 1
                continue

        out_X.append(X[start:end])
        out_y.append(target[end + offset:end + offset + target_size])
        X_index.append(index[start])                 # Keep original X_index format
        y_index.append(index[end + offset])          # Keep original y_index format

        if return_window_index:
            window_index.append(index[start:end])
        if return_center_index:
            center_index.append(index[start + window_size // 2])

        start += step_size

    result = [
        np.array(out_X),
        np.array(out_y),
        np.array(X_index),
        np.array(y_index)
    ]

    if return_window_index:
        result.append(np.array(window_index))
    if return_center_index:
        result.append(np.array(center_index))

    return tuple(result)



def time_segments_aggregate(X, interval, time_column, method=['mean']):
    """Aggregate values over given time span.

    Args:
        X (ndarray or pandas.DataFrame):
            N-dimensional sequence of values.
        interval (int):
            Integer denoting time span to compute aggregation of.
        time_column (int):
            Column of X that contains time values.
        method (str or list):
            Optional. String describing aggregation method or list of strings describing multiple
            aggregation methods. If not given, `mean` is used.

    Returns:
        ndarray, ndarray:
            * Sequence of aggregated values, one column for each aggregation method.
            * Sequence of index values (first index of each aggregated segment).
    """
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    X = X.sort_values(time_column).set_index(time_column)

    if isinstance(method, str):
        method = [method]

    start_ts = X.index.values[0]
    max_ts = X.index.values[-1]

    values = list()
    index = list()
    while start_ts <= max_ts:
        end_ts = start_ts + interval
        subset = X.loc[start_ts:end_ts - 1]
        aggregated = [
            getattr(subset, agg)(skipna=True).values
            for agg in method
        ]
        values.append(np.concatenate(aggregated))
        index.append(start_ts)
        start_ts = end_ts

    return np.asarray(values), np.asarray(index)


def cutoff_window_sequences(X, timeseries, window_size, cutoff_time=None, time_index=None):
    """Extract timeseries sequences based on cutoff times.

    Args:
        X (pandas.DataFrame):
            ``pandas.DataFrame`` containing the cutoff time alongside any other values
            that need to be used to filter the matching timeseries data.
            The cutoff time can either be set as the DataFrame index or as a column.
        timeseries (pandas.DataFrame):
            ``pandas.DataFrame`` containing the actual timeseries data. The time index
            and either be set as the DataFrame index or as a column.
        window_size (int, str or Timedelta):
            If an integer is passed, it is the number of elements to take before the
            cutoff time for each sequence. If a string or a Timedelta object is passed,
            it is the period of time we take the elements from.
        cutoff_time (str):
            Optional. If given, the indicated column will be used as the cutoff time.
            Otherwise, the table index will be used.
        time_index (str):
            Optional. If given, the indicated column will be used as the timeseries index.
            Otherwise, the table index will be used.

    Returns:
        numpy.ndarray:
            Numpy array with three dimentions. The frst dimension will have the same
            length as ``X``, and each of the 2D matrices within it will correspond to
            one row in the ``X`` table.
    """

    if cutoff_time:
        X = X.set_index(cutoff_time)

    if time_index:
        timeseries = timeseries.set_index(time_index)

    columns = list(X.columns)

    if not isinstance(window_size, int):
        window_size = pd.to_timedelta(window_size)

    output = list()
    for idx, row in enumerate(X.itertuples()):
        selected = timeseries[timeseries.index < row.Index]

        mask = [True] * len(selected)
        for column in columns:
            mask &= selected.pop(column) == getattr(row, column)

        selected = selected[mask]

        if not isinstance(window_size, int):
            min_time = selected.index[-1] - window_size
            selected = selected.loc[selected.index > min_time]
        else:
            selected = selected.iloc[-window_size:]

        len_selected = len(selected)
        if (len_selected != window_size):
            warnings.warn((
                'Sequence shorter than window_size found: {} < {}. '
                'Output shape is likely to be invalid.'
            ).format(len_selected, window_size))

        output.append(selected.values)

    output = np.array(output, dtype=object)
    if output.ndim >= 2:
        output = output.astype(float)

    return output




def slice_array_by_dims(X, target_index, axis=0):
    """Slice an ndarray by given dimensions.

    This function creates a copy of X then slices the array from the specificed
    dimension and returns the sliced array of the specified index.

    Args:
        X (ndarray):
            Array of input sequence.
        target_index (int or list[int]):
            Integer of the index to extract. Can be a list of integer values
            to extract multiple dimensions.
        axis (int or str):
            Optional. Axis along which to extract value. Default is ``0``.

    Returns:
        ndarray:
            Array of sliced values.
    """
    if isinstance(target_index, int):
        target_index = [target_index]

    dims = len(X.shape)
    if axis >= (dims):
        raise ValueError("Axis {} is outside the dimensions of X ({}).".format(axis, dims))

    indices = [slice(None)] * dims
    indices[axis] = target_index

    return X[tuple(indices)].copy()


def fillna(X, value=None, method=None, axis=None, limit=None, downcast=None):
    """Impute missing values.

    This function fills the missing values of the input sequence with the next/
    previous known value. If there are contigous NaN values, they will all be
    filled with the same next/previous known value.

    Args:
        X (ndarray or pandas.DataFrame):
            Array of input sequence.
        value:
            Optional. Value to use to fill holes (e.g. 0), alternately a dict/Series/DataFrame of
            values specifying which value to use for each index (for a Series) or column
            (for a DataFrame). Values not in the dict/Series/DataFrame will not be filled.
            This value cannot be a list. Default is None.
        method (str or list):
            Optional. String or list of strings describing whether to use forward or backward
            fill. pad / ffill: propagate last valid observation forward to next valid.
            backfill / bfill: use next valid observation to fill gap. Otherwise use ``None`` to
            fill with desired value. Possible values include
            ``[‘backfill’, ‘bfill’, ‘pad’, ‘ffill’, None]``. Default is None.
        axis (int or str):
            Optional. Axis along which to fill missing value. Possible values include 0 or
            "index", 1 or "columns". Default is None.
        limit (int):
            Optional. If method is specified, this is the maximum number of consecutive NaN values
            to forward/backward fill. In other words, if there is a gap with more than this number
            of consecutive NaNs, it will only be partially filled. If method is not specified,
            this is the maximum number of entries along the entire axis where NaNs will be filled.
            Must be greater than 0 if not None. Default is None.
        downcast (dict):
            Optional. A dict of item->dtype of what to downcast if possible, or the string "infer"
            which will try to downcast to an appropriate equal type (e.g. float64 to int64 if
            possible). Default is None.

    Returns:
        ndarray:
            Array of input sequence with imputed values.
    """
    if isinstance(method, str) or method is None:
        method = [method]

    if isinstance(X, np.ndarray):
        if X.ndim == 1:
            X_ = pd.Series(X)
        else:
            X_ = pd.DataFrame(X)

    else:
        X_ = X.copy()

    for fill in method:
        X_ = X_.fillna(value=value, method=fill, axis=axis, limit=limit, downcast=downcast)

    return X_.values
