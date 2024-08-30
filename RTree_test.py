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
from rtree import index


def read_parquet_df(path):
    """Read partitioned parquet df from disk

    Args:
        path (str): path to stored parquet dataset
    """
    dataset = ds.dataset(path)
    df = dataset.to_table().to_pandas()
    return df


def insert_data(index, datapath, dimensions):
    """Insert data into Rtree and rebalance

    Args:
        path (str): path to stored parquet dataset
    """
    df = read_parquet_df(datapath)
    for idx, row in df.iterrows():
        datapoint = (*row.to_list(), *row.to_list())
        index.insert(idx, datapoint)


def create_rtree(dimensions):
    """Create n-dimensional index stored on disk

    Args:
        path (str): path to stored parquet dataset
    """

    # Clear old index space

    if os.path.exists("./indexes/RTree/index.dat"):
        os.remove("./indexes/RTree/index.dat")
    if os.path.exists("./indexes/RTree/index.idx"):
        os.remove("./indexes/RTree/index.idx")

    p = index.Property()
    p.dimension = dimensions
    rtree = index.Index("./indexes/RTree/index", properties=p)

    return rtree


def get_datapoint(path, idx):
    """Get datapoint from stored dataset

    Args:
        path (str): path to stored df dataset
        idx (int): index of row
    """
    df = read_parquet_df(path)

    data = df.loc[idx].to_list()

    datapoint = (*data, *data)

    return datapoint


def search_rtree(index_path, datapoint, dimensions):
    """Get datapoint from stored index

    Args:
        index (object): path to stored index
        datapoint (tuple): datapoint
    """

    # Load index
    p = index.Property()
    p.dimension = dimensions
    rtree = index.Index(index_path, properties=p)

    # Query index
    results = rtree.intersection(datapoint)

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
        # rtree = create_rtree(dimensions)
        # insert_data(rtree, data_path, dimensions)

        # Test index performance
        t0 = time.perf_counter()
        datapoint = get_datapoint(data_path, 0)
        results = search_rtree("./indexes/RTree/index", datapoint, dimensions)
        t1 = time.perf_counter()
        print(f"Elapsed time for {item}: {str(t1 - t0)} seconds")


if __name__ == "__main__":
    main()
