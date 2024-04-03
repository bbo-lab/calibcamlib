import numpy as np
from copy import deepcopy


def numpy_collection_to_list(collection, do_copy=True):
    if do_copy:
        collection = deepcopy(collection)
    if isinstance(collection, np.ndarray):
        return collection.tolist()
    elif isinstance(collection, dict):
        for k in collection.keys():
            collection[k] = numpy_collection_to_list(collection[k], do_copy=False)
        return collection
    elif isinstance(collection, list):
        return [numpy_collection_to_list(le, do_copy=False) for le in collection]
    elif isinstance(collection, tuple):
        return [numpy_collection_to_list(le, do_copy=False) for le in collection]
    elif isinstance(collection, bytes):
        input = collection.decode().strip()
        return input
    elif hasattr(collection, 'dtype'):
        # To convert numpy float64 and int64 to float and int
        if np.issubdtype(collection.dtype, np.integer):
            return int(collection)
        elif np.issubdtype(collection.dtype, np.floating):
            return float(collection)
    return collection


def collection_to_array(input, do_copy=True):
    if do_copy:
        input = deepcopy(input)
    if isinstance(input, list):
        if np.array(input).dtype == "O":
            new_list = []
            for i in input:
                new_list.append(collection_to_array(i), do_copy=False)
            return new_list
        else:
            return np.array(input)
    elif isinstance(input, dict):
        for ky in input.keys():
            input[ky] = collection_to_array(input[ky], do_copy=False)
        return input
    elif isinstance(input, tuple):
        for ky in input.keys():
            input[ky] = collection_to_array(input[ky], do_copy=False)
        return input
    else:
        return input
