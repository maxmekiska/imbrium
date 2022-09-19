from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
    FunctionTransformer,
)

SCALER = {
    "": FunctionTransformer(lambda x: x, validate=True),
    "standard": StandardScaler(),
    "minmax": MinMaxScaler(),
    "maxabs": MaxAbsScaler(),
    "normalize": FunctionTransformer(
        lambda x: (x - x.min()) / (x.max() - x.min()), validate=True
    ),
}
