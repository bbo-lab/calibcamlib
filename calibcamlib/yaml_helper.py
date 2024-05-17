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


def collection_to_array(collection_in, do_copy=True):
    if do_copy:
        collection_in = deepcopy(collection_in)
    if isinstance(collection_in, list):
        if np.array(collection_in).dtype == "O":
            new_list = []
            for i in collection_in:
                new_list.append(collection_to_array(i))
            return new_list
        else:
            return np.array(collection_in)
    elif isinstance(collection_in, dict):
        for ky in collection_in.keys():
            collection_in[ky] = collection_to_array(collection_in[ky], do_copy=False)
        return collection_in
    elif isinstance(collection_in, tuple):
        input_new = []
        for ky in collection_in:
            input_new.append(collection_to_array(collection_in[ky], do_copy=False))
        return tuple(input_new)
    else:
        return collection_in
