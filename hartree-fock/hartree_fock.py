"""
Hartree-Fock method for solving the Schrödinger equation for H2
Based on Szabo and Ostlund, Modern Quantum Chemistry
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.special import erf

from basis_sets import phi_1s_CFG


def normalization(*args):
    norm = 1
    for i, arg in enumerate(args):
        norm *= (2 * arg / np.pi) ** (3 / 4)
    return norm


def S_uv(a, b, R):
    integral = (np.pi / (a + b)) ** (3 / 2) * np.exp(-a * b / (a + b) * R ** 2)
    return normalization(a, b) * integral


def calculate_S(phi_GF, R_A, N, STO_NG):
    S = np.zeros((N, N))
    for u in range(N):
        for v in range(N):
            R = np.linalg.norm(R_A[u] - R_A[v])
            for p in range(STO_NG):
                for q in range(STO_NG):
                    S[u, v] += phi_GF[u]["d"][p] * phi_GF[v]["d"][q] * S_uv(phi_GF[u]["a"][p], phi_GF[v]["a"][q], R)
    return S


def T_uv(a, b, R):
    integral = a * b / (a + b) * (3 - 2 * a * b / (a + b) * R ** 2) * (np.pi / (a + b)) ** (3 / 2) * np.exp(
        -a * b / (a + b) * R ** 2)
    return normalization(a, b) * integral


def calculate_T(phi_GF, R_A, N, STO_NG):
    T = np.zeros((N, N))
    for u in range(N):
        for v in range(N):
            R = np.linalg.norm(R_A[u] - R_A[v])
            for p in range(STO_NG):
                for q in range(STO_NG):
                    T[u, v] += phi_GF[u]["d"][p] * phi_GF[v]["d"][q] * T_uv(phi_GF[u]["a"][p], phi_GF[v]["a"][q], R)
    return T


def F_0(t):
    if t < 1E-6:
        return 1 - t / 3
    else:
        return (np.pi / t) ** (1 / 2) * erf(t ** (1 / 2)) / 2


def V_uv_C(a, b, R_A, R_B, R_C, Z_C):
    R_AB = np.linalg.norm(R_A - R_B)
    R_P = (a * R_A + b * R_B) / (a + b)
    R_PC = np.linalg.norm(R_P - R_C)
    integral = -2 * np.pi / (a + b) * Z_C * np.exp(-a * b / (a + b) * R_AB ** 2) * F_0((a + b) * R_PC ** 2)
    return normalization(a, b) * integral


def calculate_V(phi_GF, R_A, Z_A, N, STO_NG):
    V = np.zeros((N, N))
    for A in range(R_A.shape[0]):
        for u in range(N):
            for v in range(N):
                for p in range(STO_NG):
                    for q in range(STO_NG):
                        V[u, v] += phi_GF[u]["d"][p] * phi_GF[v]["d"][q] * V_uv_C(phi_GF[u]["a"][p], phi_GF[v]["a"][q],
                                                                                  R_A[u], R_A[v], R_A[A], Z_A[A])
    return V


def calculate_X(S, orthogonalization_procedure):
    s, U = np.linalg.eigh(S)
    if orthogonalization_procedure == "symmetric":
        return U @ np.diag(s ** (-1 / 2)) @ U.T
    elif orthogonalization_procedure == "canonical":
        return U @ np.diag(s ** (-1 / 2)) @ U.T
    else:
        raise ValueError("orthogonalization_procedure must be either symmetric or canonical")


def two_electron_integral(a, b, g, d, R_A, R_B, R_C, R_D):
    R_AB = np.linalg.norm(R_A - R_B)
    R_CD = np.linalg.norm(R_C - R_D)
    R_P = (a * R_A + b * R_B) / (a + b)
    R_Q = (g * R_C + d * R_D) / (g + d)
    R_PQ = np.linalg.norm(R_P - R_Q)
    integral = (2 * np.pi ** (5 / 2) / ((a + b) * (g + d) * (a + b + g + d) ** (1 / 2)) * np.exp(
        -a * b / (a + b) * R_AB ** 2 - g * d / (g + d) * R_CD ** 2) * F_0(
        (a + b) * (g + d) / (a + b + g + d) * R_PQ ** 2))
    return normalization(a, b, g, d) * integral


def calculate_G(phi_GF, P, R_A, N, STO_NG):
    G = np.zeros((N, N))
    for u in range(N):
        for v in range(N):
            for l in range(N):
                for s in range(N):
                    for p in range(STO_NG):
                        for q in range(STO_NG):
                            for r in range(STO_NG):
                                for t in range(STO_NG):
                                    a = phi_GF[u]["a"][p]
                                    b = phi_GF[v]["a"][q]
                                    g = phi_GF[l]["a"][r]
                                    d = phi_GF[s]["a"][t]
                                    # pcc = product of contraction coefficients
                                    pcc = phi_GF[u]["d"][p] * phi_GF[v]["d"][q] * phi_GF[l]["d"][r] * phi_GF[s]["d"][t]
                                    # uvsl = (uv|sl)
                                    uvsl = two_electron_integral(a, b, d, g, R_A[u], R_A[v], R_A[s], R_A[l])
                                    ulsv = two_electron_integral(a, g, d, b, R_A[u], R_A[l], R_A[s], R_A[v])
                                    G[u, v] += pcc * P[l, s] * (uvsl - 1 / 2 * ulsv)
    return G


def single_point_calculation(phi_GF, P, R_A, N, STO_NG, H_core, X):
    # Calculate G
    G = calculate_G(phi_GF, P, R_A, N, STO_NG)

    # Calculate Fock matrix (F)
    F = H_core + G

    # Calculate transformed Fock matrix (F_prime)
    F_prime = X.T @ F @ X

    # Diagonalize F_prime to obtain C_prime and epsilon
    epsilon, C_prime = np.linalg.eigh(F_prime)

    # Calculate C = X @ C_prime
    C = X @ C_prime
    return C, epsilon, F


def calculate_P(C, N):
    P = np.zeros((N, N))
    for u in range(N):
        for v in range(N):
            for a in range(int(N / 2)):
                P[u, v] += 2 * C[u, a] * C[v, a]
    return P


def SCF(phi_GF, R_A, Z_A, N, STO_NG, N_iter, orthogonalization_procedure, verbose=False):
    # Calculate overlap matrix (S)
    S = calculate_S(phi_GF, R_A, N, STO_NG)

    # Calculate T
    T = calculate_T(phi_GF, R_A, N, STO_NG)

    # Calculate V
    V = calculate_V(phi_GF, R_A, Z_A, N, STO_NG)

    # Calculate H_core
    H_core = T + V

    # Diagonalize S and obtain transformation matrix (X)
    X = calculate_X(S, orthogonalization_procedure)

    # Guess at the density matrix (P)
    P = np.zeros((N, N))

    # SCF loop
    dP = 1e-5
    for i in range(N_iter):
        C, epsilon, F = single_point_calculation(phi_GF, P, R_A, N, STO_NG, H_core, X)

        # Form a new P
        P_old = P.copy()
        P = calculate_P(C, N)

        # Calculate the difference between the new and old P
        P_diff = np.linalg.norm(P - P_old)
        if P_diff < dP:
            if verbose:
                print("SCF converged after {} iterations".format(i + 1))
            break

    # Calculate total electronic energy
    E_0 = 1 / 2 * np.sum(P * (H_core + F))

    # Calculate nuclear repulsion energy
    E_nuc = Z_A[0] * Z_A[1] / np.linalg.norm(R_A[0] - R_A[1])

    # Calculate total energy
    E_tot = E_0 + E_nuc

    return C, epsilon, E_0, E_tot


def main():
    # Calculate potential energy surface
    Rs = np.linspace(0.5, 5, 91)
    E_H2 = np.zeros(len(Rs))
    for i, R in enumerate(Rs):
        print("Calculating H2 at R = {} bohr".format(R))

        # Specify molecule
        R_A = np.array([[0, 0, 0], [R, 0, 0]])
        Z_A = np.array([1, 1])
        N = 2

        # Specify basis set
        z = 1.24
        STO_NG = 3
        phi_GF = [phi_1s_CFG["z"][z]["STO-NG"][STO_NG] for _ in range(N)]

        # Calculation parameters
        N_iter = 5
        orthogonalization_procedure = "symmetric"

        # Run SCF calculation
        C, epsilon, E_0, E_tot = SCF(phi_GF, R_A, Z_A, N, STO_NG, N_iter, orthogonalization_procedure, verbose=False)

        # Store energy
        E_H2[i] = E_tot

    # Plot potential energy surface
    E_H = -0.4666
    plt.plot(Rs, E_H2 - 2 * E_H)
    plt.xlabel("R (bohr)")
    plt.ylabel(r"$E({\rm H_2})-2E({\rm H})$ (hartree)")
    plt.tight_layout()
    plt.savefig("H2.png")


if __name__ == "__main__":
    main()
