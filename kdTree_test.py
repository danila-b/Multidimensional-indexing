from random import seed
from indexes.k_d_Tree.kdTree import kdtree, search_kdtree
import time
import numpy
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from datetime import datetime
import pickle
import os


def read_parquet_df(path):
    """Read partitioned parquet df from disk

    Args:
        path (str): path to stored parquet dataset
    """
    dataset = ds.dataset(path)
    df = dataset.to_table().to_pandas()
    return df


def generate_rand_int_dataframe(
    dimensions: int = 10,
    rows: int = 1000000,
    partition_thresh: int = 100000,
    data_variety: int = 1,
    path: str = "data/test/int_dataset",
):
    """Generate int parquet dataframe of the given size and store it on disk

    Args:
        dimensions (int): number of dataset dimension
        rows (int): number od dataset rows
        partition_thresh (int): number of rows in each partition
        data_variety (int): generated data variety scale
        path (str): path to store dataset
    """
    # Generate dataframe
    df = pd.DataFrame(
        np.random.randint(0, 100 * data_variety, size=(rows, dimensions)),
        columns=[str(i) for i in range(1, dimensions + 1)],
    )

    # Add partition column
    df["part_col"] = list(map(lambda x: x // partition_thresh, df.index.tolist()))

    # Convert from pandas to Arrow
    table = pa.Table.from_pandas(df)

    pq.write_to_dataset(
        table,
        path,
        partition_filename_cb=lambda x: "chunk_file_" + str(x[0]) + ".parquet",
        partition_cols=["part_col"],
    )


def create_kd_tree(data_path, leafsize):
    """Creates kdTree index from given parquet dataframe

    Args:
        data_path (str): path to parquet dataframe
        leafsize (int): size of the kdTree nodes
    """
    df = read_parquet_df(data_path)
    data = df.to_numpy()
    kd_tree = kdtree(data, leafsize)
    return kd_tree


def store_kdTree(kd_tree, path):
    """Store kdTree index and data on disk

    Args:
        path (str): path to store file
        kd_tree (list or dict): kdTree structure and data
    """
    with open(path, "wb") as fp:
        pickle.dump(kd_tree, fp)


def get_datapoint(path, idx, leafsize):
    """Get datapoint from stored dataset

    Args:
        path (str): path to stored df dataset
        idx (int): index of row
    """
    df = read_parquet_df(path)

    data = df.to_numpy()

    ndata = data.shape[1]
    param = data.shape[0]
    datapoint = data[:, idx].reshape((param, 1)).repeat(leafsize, axis=1)

    return datapoint


def search_kdTree(index_path, datapoint):
    """Load kdTree from disk and search datapoint

    Args:
        index_path (str): path to stored index
        datapoint (list): datapoint
    """
    datapoint = datapoint

    with open(index_path, "rb") as fp:
        kdTree = pickle.load(fp)

    search_data = search_kdtree(kdTree, datapoint, 1)


def main():

    # For online data generation

    # data_path = "data/test/int_dataset"
    # data_variety = 10
    # dimensions = 10
    # rows = 1000000
    # partition_thresh = 100000

    # # Create a dataset of int`s
    # generate_rand_int_dataframe(
    #     dimensions=dimensions,
    #     rows=rows,
    #     partition_thresh=partition_thresh,
    #     data_variety=data_variety,
    #     path=data_path,
    # )

    # Get datasets
    path = "data/test/"
    datasets = os.listdir(path)

    for item in datasets:
        data_path = path + item

        # Create index
        index_path = "indexes/k_d_Tree/index_data"
        leafsize = 100
        kdTree = create_kd_tree(data_path, leafsize)
        store_kdTree(kdTree, index_path)

        # Test index performance
        t0 = time.perf_counter()
        datapoint = get_datapoint(data_path, 0, leafsize)
        search_kdTree(index_path, datapoint)
        t1 = time.perf_counter()
        print(f"Elapsed time for {item}: {str(t1 - t0)} seconds")


if __name__ == "__main__":
    main()
