from typing import Tuple

from numpy import array, dstack, empty, vstack


def data_prep_uni(data: array) -> array:
    """Prepares data for model intake.
    Parameters:
        data (array): Input time series.
    Returns:
        data (array): Input time series.
    """
    data = data.reshape(-1, 1)

    return data


def data_prep_multi(target: array, features: array) -> array:
    """Prepare target and feature numpy arrays for mulitvariate model intake.
    Parameters:
        target (array): Array containing multi-feature data.
        features (array): All features that should be considered.
    Returns:
        data (array): Array containing sequences of selected features.
    """
    if target.ndim == 2:
        target = target.T
    data = vstack((target, features.T))

    return data


def sequence_prep_standard_uni(
    input_sequence: array, steps_past: int, steps_future: int
) -> [(array, array)]:
    """Prepares data input into X and y sequences. Length of the X sequence
    is determined by steps_past while the length of y is determined by
    steps_future. In detail, the predictor looks at sequence X and
    predicts sequence y.
     Parameters:
         input_sequence (array): Sequence that contains time series in
         array format
         steps_past (int): Steps the predictor will look backward
         steps_future (int): Steps the predictor will look forward
     Returns:
         X (array): Array containing all looking back sequences
         y (array): Array containing all looking forward sequences
    """
    length = len(input_sequence)
    if length == 0:
        return (empty(shape=[steps_past, steps_past]), 0)
    X = []
    y = []
    if length <= steps_past:
        raise ValueError(
            "Input sequence is equal to or shorter than steps to look backwards"
        )
    if steps_future <= 0:
        raise ValueError("Steps in the future need to be bigger than 0")

    for i in range(length):
        last = i + steps_past
        if last > length - steps_future:
            break
        X.append(input_sequence[i:last])
        y.append(input_sequence[last : last + steps_future])
    y = array(y)
    X = array(X)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y


def sequence_prep_standard_multi(
    input_sequence: array, steps_past: int, steps_future: int
) -> [(array, array)]:
    """Prepares data input into X and y sequences. Lenght of the X sequence is dertermined by steps_past while the length of y is determined by steps_future. In detail, the predictor looks at sequence X and predicts sequence y.
    Parameters:
        input_sequence (array): Sequence that contains time series in array format
        steps_past (int): Steps the predictor will look backward
        steps_future (int): Steps the predictor will look forward
    Returns:
        X (array): Array containing all looking back sequences
        y (array): Array containing all looking forward sequences
    """
    length = len(input_sequence)
    if length == 0:
        return (empty(shape=[steps_past, steps_past]), 0)
    X = []
    y = []
    if length <= steps_past:
        raise ValueError(
            "Input sequence is equal to or shorter than steps to look backwards"
        )
    if steps_future <= 0:
        raise ValueError("Steps in the future need to be bigger than 0")

    for i in range(length):
        last = i + steps_past
        if last > length - steps_future:
            X.append(input_sequence[i:last])
            y.append(input_sequence[last - 1 : last - 1 + steps_future])
            break
        X.append(input_sequence[i:last])
        # modification to use correct target y sequence
        y.append(input_sequence[last - 1 : last - 1 + steps_future])
    y = array(y)
    X = array(X)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y


def sequence_prep_hybrid_uni(
    input_sequence: array, sub_seq: int, steps_past: int, steps_future: int
) -> [(array, array, int)]:
    """Prepares data input into X and y sequences. Length of the X sequence
    is determined by steps_past while the length of y is determined by
    steps_future. In detail, the predictor looks at sequence X and
    predicts sequence y.
     Parameters:
         input_sequence (array): Sequence that contains time series in
         array format.
         sub_seq (int): Further division of given steps a predictor will
         look backward.
         steps_past (int): Steps the predictor will look backward.
         steps_future (int): Steps the predictor will look forward.
     Returns:
         X (array): Array containing all looking back sequences.
         y (array): Array containing all looking forward sequences.
         modified_back (int): Modified looking back sequence length.
    """
    length = len(input_sequence)
    if length == 0:
        return (0, 0, steps_past // sub_seq)
    X = []
    y = []
    if length <= steps_past:
        raise ValueError(
            "Input sequence is equal to or shorter than steps to look backwards"
        )
    if steps_future <= 0:
        raise ValueError("Steps in the future need to be bigger than 0")

    for i in range(length):
        last = i + steps_past
        if last > length - steps_future:
            break
        X.append(input_sequence[i:last])
        y.append(input_sequence[last : last + steps_future])
    y = array(y)
    X = array(X)
    modified_back = X.shape[1] // sub_seq
    X = X.reshape((X.shape[0], sub_seq, modified_back, 1))
    return X, y, modified_back


def sequence_prep_hybrid_multi(
    input_sequence: array, sub_seq: int, steps_past: int, steps_future: int
) -> [(array, array, int)]:
    """Prepares data input into X and y sequences. Lenght of the X sequence
    is dertermined by steps_past while the length of y is determined by
    steps_future. In detail, the predictor looks at sequence X and
    predicts sequence y.
         Parameters:
             input_sequence (array): Sequence that contains time series in
             array format
             sub_seq (int): Division into subsequences.
             steps_past (int): Steps the predictor will look backward
             steps_future (int): Steps the predictor will look forward
         Returns:
             X (array): Array containing all looking back sequences
             y (array): Array containing all looking forward sequences
    """
    length = len(input_sequence)
    if length == 0:
        return (empty(shape=[steps_past, steps_past]), 0)
    X = []
    y = []
    if length <= steps_past:
        raise ValueError(
            "Input sequence is equal to or shorter than steps to look backwards"
        )
    if steps_future <= 0:
        raise ValueError("Steps in the future need to be bigger than 0")

    for i in range(length):
        last = i + steps_past
        if last > length - steps_future:
            X.append(input_sequence[i:last])
            y.append(input_sequence[last - 1 : last - 1 + steps_future])
            break
        X.append(input_sequence[i:last])
        # modification to use correct target y sequence
        y.append(input_sequence[last - 1 : last - 1 + steps_future])
    y = array(y)
    X = array(X)
    modified_back = X.shape[1] // sub_seq
    X = X.reshape((X.shape[0], sub_seq, modified_back, 1))
    return (
        X,
        y,
        modified_back,
    )  # special treatment to account for sub sequence division


def multistep_prep_standard(
    input_sequence: array, steps_past: int, steps_future: int
) -> [(array, array)]:
    """This function prepares input sequences into a suitable input format for a multivariate multistep model. The first seqeunce in the array needs to be the target variable y.
    Parameters:
        input_sequence (array): Sequence that contains time series in array format
        steps_past (int): Steps the predictor will look backward
        steps_future (int): Steps the predictor will look forward
    Returns:
        X (array): Array containing all looking back sequences
        y (array): Array containing all looking forward sequences
    """
    X = []
    Y = []
    for i in range(len(input_sequence)):
        if i == 0:  # target variable should be first sequence
            _, y = sequence_prep_standard_multi(
                input_sequence[0], steps_past, steps_future
            )
            Y.append(y)
            continue  # skip since target column not requiered in X array
        x, _ = sequence_prep_standard_multi(input_sequence[i], steps_past, steps_future)
        X.append(x)
    X = dstack(X)
    Y = Y[0]  # getting array out of list
    return X, Y


def multistep_prep_hybrid(
    input_sequence: array, sub_seq: int, steps_past: int, steps_future: int
) -> [(array, array)]:
    """This function prepares input sequences into a suitable input format
    for a multivariate multistep model. The first seqeunce in the array
    needs to be the target variable y.
         Parameters:
             input_sequence (array): Sequence that contains time series
             in array format
             sub_seq (int): Devision into subsequences.
             steps_past (int): Steps the predictor will look backward
             steps_future (int): Steps the predictor will look forward
         Returns:
             X (array): Array containing all looking back sequences
             y (array): Array containing all looking forward sequences
    """
    X = []
    Y = []
    for i in range(len(input_sequence)):
        if i == 0:  # target variable should be first sequence
            _, y, _ = sequence_prep_hybrid_multi(
                input_sequence[0], sub_seq, steps_past, steps_future
            )
            Y.append(y)
            continue  # skip since target column not requiered in X array
        x, _, mod = sequence_prep_hybrid_multi(
            input_sequence[i], sub_seq, steps_past, steps_future
        )
        X.append(x)
    X = dstack(X)
    Y = Y[0]  # getting array out of list
    return X, Y, mod


def train_test_split(data: array, test_size=0.2) -> Tuple[array, array]:
    """Splits the time series data into training and testing sets."""
    n_samples = len(data)
    n_test = int(test_size * n_samples)

    train_data = data[:-n_test]
    test_data = data[-n_test:]

    return train_data, test_data
