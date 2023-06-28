""" Contains functions for calculating neighbor lists """
import numpy as np


def sc(N, a):
    r = np.zeros((N ** 3, 3))
    i = 0
    for j in range(N):
        for k in range(N):
            for l in range(N):
                r[i] = np.array([j, k, l]) * a
                i += 1
    return r


def neighbor_list(r, a, r_cut=7.5, padding=1.5):
    nl = {}
    cell = len(r) * a
    for i in range(len(r)):
        nl[i] = []
        for j in range(len(r)):
            if i != j:
                dx = r[i, 0] - r[j, 0] - np.rint((r[i, 0] - r[j, 0]) / cell) * cell
                dy = r[i, 1] - r[j, 1] - np.rint((r[i, 1] - r[j, 1]) / cell) * cell
                dz = r[i, 2] - r[j, 2] - np.rint((r[i, 2] - r[j, 2]) / cell) * cell
                dij = np.linalg.norm(np.array([dx, dy, dz]))
                if dij < r_cut * padding:
                    nl[i].append(j)
    return nl


def main():
    N = 10
    a = 3.4
    r = sc(N, a)
    nl = neighbor_list(r, a)
    print(nl)


if __name__ == "__main__":
    main()
