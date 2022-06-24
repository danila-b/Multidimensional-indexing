from random import randrange, randint
import numpy as np
from hilbert import decode, encode
import time
import os
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds

# # Turn an ndarray of Hilber integers into locations.
# # 2 is the number of dimensions, 3 is the number of bits per dimension
# locs = decode(np.array([[1, 2, 3], [4, 5, 6]]), 10, 3)

# print(locs)
# # prints [[0 1]
# #         [1 1]
# #         [1 0]]

# # You can go the other way also, of course.
# H = encode(locs, 10, 3)

# print(H)
# # prints array([1, 2, 3], dtype=uint64)


def read_parquet_df(path):
    """Read partitioned parquet df from disk

    Args:
        path (str): path to stored parquet dataset
    """
    dataset = ds.dataset(path)
    df = dataset.to_table().to_pandas()
    return df


def create_hilbert(data_path, dimensions):
    """Creates dimensional Hilbert grid

    Args:
        data_path (str): row boundaries range
        dimensions (int): dimensionality
    """

    df = read_parquet_df(data_path)
    df = df.to_numpy()
    # df = df[:100]

    hilbert = decode(df, dimensions, 3)

    return hilbert


def store_index(hilbert, path):
    """Store Hilbert curve

    Args:
        data_path (str): row boundaries range
        dimensions (int): dimensionality
    """
    with open(path, "wb") as fp:
        pickle.dump(hilbert, fp)


def search_hilbert(hilbert, dimensions, datapoint):
    """Search dimensional Hilbert index

    Args:
        zi (object): zorder index
        datapoint (int): datapoint
    """

    results = encode(hilbert, dimensions, 3)

    return results


def main():

    # Get datasets
    path = "data/test/"
    datasets = os.listdir(path)

    datasets = [datasets[0]]  # TODO: remove
    print(datasets)

    for item in datasets:
        data_path = path + item
        dimensions = int(str(item)[12:14])

        # Create index
        hilbert = create_hilbert(data_path, dimensions)

        print(hilbert)

        # Test index performance
        t0 = time.perf_counter()
        datapoint = randint(1, 10000)
        results = search_hilbert(hilbert, dimensions, datapoint)
        t1 = time.perf_counter()
        print(f"Elapsed time for {item}: {str(t1 - t0)} seconds")


if __name__ == "__main__":
    main()
