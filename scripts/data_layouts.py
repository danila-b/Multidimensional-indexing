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
    path: str = "test/int_dataset",
):

    # Generate dataframe
    df = pd.DataFrame(
        np.random.randint(0, 100, size=(rows, dimensions)),
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


def main():
    return 0
    # Delete old data

    # generate_rand_int_dataframe()

    # # Read partitioned dataset
    # dataset = ds.dataset("test/int_dataset")

    # df = dataset.to_table().to_pandas()
    # print(df)


if __name__ == "__main__":
    main()
