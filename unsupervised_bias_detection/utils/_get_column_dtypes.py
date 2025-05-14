import numpy as np
import pandas as pd


def get_column_dtypes(data) -> dict:
    """
    Return a dictionary mapping column names to abstract data types that are compatible with the processor.
    
    The mapping is as follows:
    - float64, float32, int64, int32 -> "numerical"
    - bool -> "boolean"
    - datetime64[...] -> "datetime"
    - timedelta64[...] -> "timedelta"
    - All others (e.g., object) -> "categorical"
    """
    def map_dtype(dtype: str) -> str:
        if dtype in ['float64', 'float32', 'int64', 'int32']:
            return "numerical"
        elif dtype == 'bool':
            return "boolean"
        elif 'datetime' in dtype:
            return "datetime"
        elif 'timedelta' in dtype:
            return "timedelta"
        else:
            return "categorical"
    
    if isinstance(data, pd.DataFrame):
        return {col: map_dtype(str(dtype)) for col, dtype in data.dtypes.items()}
    elif isinstance(data, np.ndarray) and data.dtype.names is not None:
        return {name: map_dtype(str(data.dtype.fields[name][0])) for name in data.dtype.names}
    else:
        raise TypeError("Data must be a pandas DataFrame or a structured numpy array.")