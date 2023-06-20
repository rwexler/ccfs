""" Hartree-Fock method for solving the Schrödinger equation for H2"""

import numpy as np
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


def main(orthogonalization_procedure="symmetric"):
    # Specify molecule
    R_A = np.array([[0, 0, 0], [1.4, 0, 0]])
    Z_A = np.array([1, 1])
    N = 2

    # Specify basis set
    z = 1.24
    STO_NG = 3
    phi_GF = [phi_1s_CFG["z"][z]["STO-NG"][STO_NG] for _ in range(N)]

    # Guess at the density matrix (P)
    P = np.zeros((N, N))

    # Calculate overlap matrix (S)
    S = calculate_S(phi_GF, R_A, N, STO_NG)

    # Calculate T
    T = calculate_T(phi_GF, R_A, N, STO_NG)

    # Calculate V
    V = np.zeros((N, N))
    for A in range(R_A.shape[0]):
        for u in range(N):
            for v in range(N):
                for p in range(STO_NG):
                    for q in range(STO_NG):
                        V[u, v] += phi_GF[u]["d"][p] * phi_GF[v]["d"][q] * V_uv_C(phi_GF[u]["a"][p], phi_GF[v]["a"][q],
                                                                                  R_A[u], R_A[v], R_A[A], Z_A[A])

    # Calculate H_core
    H_core = T + V

    # Diagonalize S and obtain transformation matrix (X)
    X = calculate_X(S, orthogonalization_procedure)

    # Calculate G
    G = calculate_G(phi_GF, P, R_A, N, STO_NG)

    # Calculate Fock matrix (F)
    F = H_core + G

    # Calculate transformed Fock matrix (F_prime)
    F_prime = X.T @ F @ X

    # Diagonalize F_prime to obtain C_prime and epsilon
    epsilon, C_prime = np.linalg.eigh(F_prime)
    print(epsilon)
    print(C_prime)


if __name__ == "__main__":
    main(orthogonalization_procedure="symmetric")
