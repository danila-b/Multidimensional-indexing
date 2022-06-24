import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from datetime import datetime

# Generate dataframe and store it as partitioned Parquet to path
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


def generate_datasets():

    for rows in [1000000, 5000000, 10000000]:
        for dimensions in [10, 15, 20, 50, 100]:
            data_path = f"data/test/int_dataset_{dimensions}_{rows/1000000}M"
            data_variety = 100
            dimensions = dimensions
            rows = rows
            partition_thresh = 1000000

            # Create a dataset of int`s
            generate_rand_int_dataframe(
                dimensions=dimensions,
                rows=rows,
                partition_thresh=partition_thresh,
                data_variety=data_variety,
                path=data_path,
            )


def main():
    generate_datasets()


if __name__ == "__main__":
    main()
