import pickle
import concurrent.futures
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Any

def save_pickle(output_path, obj):
    with open(output_path, "wb") as f:
        pickle.dump(obj, f, protocol=-1)

def save_multi_pickles(path: str, files: List[Tuple[str, Any]]):
    with ThreadPoolExecutor(len(files)) as executor:
        futures = []
        for file in files:
            future = executor.submit(save_pickle, os.path.join(path, file[0]), file[1])
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            pass

def load_pickle(input_path):
    obj = None
    with open(input_path, "rb") as f:
        obj = pickle.load(f)
    return obj

def load_multi_pickles(input_path_list):
    objs = []
    with ThreadPoolExecutor(len(input_path_list)) as executor:
        futures = []
        for input_path in input_path_list:
            future = executor.submit(load_pickle, input_path)
            futures.append(future)

        for future in futures:
            objs.append(future.result())

    return objs