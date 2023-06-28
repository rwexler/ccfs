""" Molecular statics simulation of argon """

import numpy as np

rng = np.random.default_rng()


def u_lj(r, epsilon=0.010323, sigma=3.405, r_tail=7, r_cut=7.5, a=-6.8102128E-3, b=-5.5640876E-3):
    """ Lennard-Jones potential energy """
    u = None
    if r > r_cut:
        u = 0
    elif r > r_tail:
        u = a * (r - r_cut) ** 3 + b * (r - r_cut) ** 2
    else:
        u = 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)
    if u is None:
        raise ValueError("u is None")
    return u


def f_lj(r, epsilon=0.010323, sigma=3.405, r_tail=7, r_cut=7.5, a=-6.8102128E-3, b=-5.5640876E-3):
    """ Lennard-Jones force """
    f = None
    if r > r_cut:
        f = 0
    elif r > r_tail:
        f = 3 * a * (r - r_cut) ** 2 + 2 * b * (r - r_cut)
    else:
        f = 24 * epsilon * (-2 * (sigma / r) ** 12 + (sigma / r) ** 6) / r
    if f is None:
        raise ValueError("f is None")
    return f


def simple_cubic(n, a):
    """ Simple cubic lattice """
    r = np.zeros((n ** 3, 3))
    i = 0
    for i_x in range(n):
        for i_y in range(n):
            for i_z in range(n):
                r[i] = np.array([i_x, i_y, i_z]) * a
                i += 1
    return r


def write_xyz(r, filename):
    """ Write positions to xyz file """
    with open(filename, "w") as f:
        f.write(str(len(r)) + "\n")
        f.write("\n")
        for i in range(len(r)):
            f.write("Ar " + str(r[i, 0]) + " " + str(r[i, 1]) + " " + str(r[i, 2]) + "\n")


def get_neighbors(i, r, cell_length, r_cut=7.5, padding=1.5):
    """ Get neighbors of atom i """
    neighbors = []
    for j in range(len(r)):
        if i == j:
            continue
        rx, ry, rz = rij(r[i], r[j], cell_length)
        d = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
        if d < r_cut * padding:
            neighbors.append(j)
    return neighbors


def get_neighbor_list(r, cell_length, r_cut=7.5, padding=1.5):
    """ Get neighbor list """
    neighbor_list = []
    for i in range(len(r)):
        neighbor_list.append(get_neighbors(i, r, cell_length, r_cut, padding))
    return neighbor_list


def rij(ri, rj, cell_length):
    """ Vector from atom j to atom i """
    rx = ri[0] - rj[0] - round((ri[0] - rj[0]) / cell_length) * cell_length
    ry = ri[1] - rj[1] - round((ri[1] - rj[1]) / cell_length) * cell_length
    rz = ri[2] - rj[2] - round((ri[2] - rj[2]) / cell_length) * cell_length
    return np.array([rx, ry, rz])


def calculate_pair_force(ri, rj, cell_length):
    """ Force on atom i due to atom j """
    rx, ry, rz = rij(ri, rj, cell_length)
    d = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
    return f_lj(d) * np.array([rx, ry, rz]) / d


def calculate_force(i, r, neighbor_list, cell_length):
    """ Force on atom i """
    f = np.zeros(3)
    for j in neighbor_list[i]:
        f += calculate_pair_force(r[i], r[j], cell_length)
    return f


def calculate_forces(r, neighbor_list, cell_length):
    """ Forces on all atoms """
    f = np.zeros((len(r), 3))
    for i in range(len(r)):
        f[i] = calculate_force(i, r, neighbor_list, cell_length)
    return f


def calculate_pair_energy(ri, rj, cell_length):
    """ Potential energy of atom i due to interaction with atom j """
    rx, ry, rz = rij(ri, rj, cell_length)
    d = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
    return u_lj(d)


def calculate_particle_energy(i, r, neighbor_list, cell_length):
    """ Potential energy of atom i """
    E = 0
    for j in neighbor_list[i]:
        E += calculate_pair_energy(r[i], r[j], cell_length)
    return E


def calculate_energy(r, neighbor_list, cell_length):
    """ Total potential energy """
    E = 0
    for i in range(len(r)):
        E += calculate_particle_energy(i, r, neighbor_list, cell_length)
    return E / 2


def random_positions(n, cell_length):
    """ Random positions """
    r = np.zeros((n, 3))
    for i in range(n):
        r[i] = rng.normal(cell_length / 2, cell_length / 8, 3)
    return r


def write_extxyz(r, cell_length, filename):
    """ Write positions to extended xyz file """
    with open(filename, "w") as f:
        f.write(str(len(r)) + "\n")
        f.write("Lattice=\"" + str(cell_length) + " 0.0 0.0 0.0 " + str(cell_length) + " 0.0 0.0 0.0 " + str(
            cell_length) + "\" Properties=species:S:1:pos:R:3 Time=0\n")
        for i in range(len(r)):
            f.write("Ar " + str(r[i, 0]) + " " + str(r[i, 1]) + " " + str(r[i, 2]) + "\n")


def main():
    # structure parameters
    n = 4  # number of particles
    cell_length = 20  # length of the cell
    r = random_positions(n, cell_length)
    nl = get_neighbor_list(r, cell_length)  # neighbor list

    # simulation parameters
    n_steps = 100000  # number of steps
    f_threshold = 1e-4  # threshold for convergence
    step_size = 0.05  # step size
    io_interval = 100  # interval for writing output
    nl_interval = 10  # interval for updating neighbor list

    # io
    write_extxyz(r, cell_length, "initial.extxyz")
    output = open("output.txt", "w")
    output.write("Step\tMaximum force\tEnergy\n")

    # main loop
    for step in range(n_steps):
        f = calculate_forces(r, nl, cell_length)
        E = calculate_energy(r, nl, cell_length)
        f_max = np.max(np.sqrt(np.sum(f ** 2, axis=1)))
        if f_max < f_threshold:
            output.write("\n")
            output.write("Converged in " + str(step) + " steps\n")
            output.write("Maximum force: " + str(f_max) + "\n")
            output.write("Energy: " + str(E) + "\n")
            break
        if step % io_interval == 0:
            output.write(f"{step}\t{f_max:.10f}\t{E:.10f}\n")
        if step % nl_interval == 0:
            nl = get_neighbor_list(r, cell_length)
        r -= f * step_size

    write_extxyz(r, cell_length, "final.extxyz")


if __name__ == "__main__":
    main()
