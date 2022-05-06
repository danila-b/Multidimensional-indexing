from rtree import index
import time


def main():
    # Create n-dimensional index stored on disk
    p = index.Property()
    p.dimension = 4
    idx3d = index.Index("./indexes/RTree/3d_index", properties=p)
    idx3d.insert(1, (0, 60, 23.0, 1.0, 0, 60, 42.0, 2.0))
    values = idx3d.intersection((-1, 62, 22, 1, -1, 62, 40, 1))

    for item in values:
        print(item)


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    t1 = time.perf_counter()
    print(f"Elapsed time {str(t1 - t0)} seconds")
