import copy
import numpy as np
from copy import deepcopy


def numpy_collection_to_list(collection):
    if isinstance(collection, np.ndarray):
        return collection.tolist()
    elif isinstance(collection, dict):
        for k in collection.keys():
            collection[k] = numpy_collection_to_list(collection[k])
        return collection
    elif isinstance(collection, list):
        return [numpy_collection_to_list(le) for le in collection]
    elif isinstance(collection, tuple):
        return tuple(numpy_collection_to_list(le) for le in collection)
    elif hasattr(collection, 'dtype'):
        # To convert numpy float64 and int64 to float and int
        if np.issubdtype(collection.dtype, np.integer):
            return int(collection)
        elif np.issubdtype(collection.dtype, np.floating):
            return float(collection)

    return deepcopy(collection)


def collection_to_array(input):
    input = copy.deepcopy(input)
    if isinstance(input, list):
        if np.array(input).dtype == "O":
            new_list = []
            for i in input:
                new_list.append(collection_to_array(i))
            return new_list
        else:
            return np.array(input)
    elif isinstance(input, dict):
        for ky in input.keys():
            input[ky] = collection_to_array(input[ky])
        return input
    else:
        return input
