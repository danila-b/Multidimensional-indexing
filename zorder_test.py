from indexes.Zorder.Zorderindex import ZOrderIndexer
import time
import os
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds


def read_parquet_df(path):
    """Read partitioned parquet df from disk

    Args:
        path (str): path to stored parquet dataset
    """
    dataset = ds.dataset(path)
    df = dataset.to_table().to_pandas()
    return df


def create_zorder(data_path, dimensions):
    """Creates dimensional Zorder grid

    Args:
        data_path (str): row boundaries range
        dimensions (int): dimensionality
    """

    df = read_parquet_df(data_path)

    range_max = int(df.to_numpy().max())
    range_min = int(df.to_numpy().min())

    zi = ZOrderIndexer((range_min, range_max), (range_min, range_max))

    return zi


def store_index(zorder, path):
    """Creates dimensional Zorder grid

    Args:
        data_path (str): row boundaries range
        dimensions (int): dimensionality
    """
    with open(path, "wb") as fp:
        pickle.dump(zorder, fp)


def search_zorder(zi, datapoint):
    """Creates dimensional Zorder grid

    Args:
        zi (object): zorder index
        datapoint (int): datapoint
    """

    results = zi.next_zorder_index(datapoint)

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
        zi = create_zorder(data_path, dimensions)

        # Test index performance
        t0 = time.perf_counter()
        results = search_zorder(zi, 25)
        print(results)
        t1 = time.perf_counter()
        print(f"Elapsed time for {item}: {str(t1 - t0)} seconds")


if __name__ == "__main__":
    main()
