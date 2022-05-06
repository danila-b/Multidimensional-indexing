from Zorderindex import ZOrderIndexer
import time

# TODO: Solve the problem of creating N-dimensional index (find solution for n-range bit interleaving, because articles has only 2-dimensonal case or naive Z-order concatenation)


def main():
    # Create 2-dimensional z-order index

    zi = ZOrderIndexer((1, 1000), (2, 60))

    z_2_2 = zi.zindex(2, 2)

    print(zi.next_zorder_index(z_2_2))
    print(zi.next_zorder_index(15))


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    t1 = time.perf_counter()
    print(f"Elapsed time {str(t1 - t0)} seconds")
