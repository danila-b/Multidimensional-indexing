from random import seed
from kdTree import kdtree, search_kdtree
import time
import numpy


def main():
    K = 11
    ndata = 100000
    ndim = 100
    numpy.random.seed(10)
    data = 10 * numpy.random.rand(ndata * ndim).reshape((ndim, ndata))

    # knn_search(data, K)
    leafsize = 1
    kd_tree = kdtree(data, leafsize)

    ndata = data.shape[1]
    param = data.shape[0]
    datapoint = data[:, 1].reshape((param, 1)).repeat(leafsize, axis=1)
    # print(data)
    # print(datapoint)
    # print(kd_tree)

    print(search_kdtree(kd_tree, datapoint, 4))


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    t1 = time.perf_counter()
    print(f"Elapsed time {str(t1 - t0)} seconds")
