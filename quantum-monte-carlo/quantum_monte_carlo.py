"""
Variational Monte Carlo for solving the harmonic oscillator in one dimension
Based on Thijssen, Computational Physics
"""

import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(742023)


def psi_T(x, a):
    """ Trial wave function """
    return np.exp(-a * x ** 2)


def psi_T_second_derivative(x, a):
    """ Second derivative of the trial wave function """
    return 2 * a * np.exp(-a * x ** 2) * (2 * a * x ** 2 - 1)


def H_psi_T(x, a):
    """ Hamiltonian acting on the trial wave function """
    return -0.5 * psi_T_second_derivative(x, a) + 0.5 * x ** 2 * psi_T(x, a)


def E_L(x, a):
    """ Local energy """
    return H_psi_T(x, a) / psi_T(x, a)


def plots():
    x = np.linspace(-5, 5, 1000)
    a = 0.5
    plt.plot(x, psi_T(x, a), label=r"$\psi_T(x)$")
    plt.plot(x, psi_T_second_derivative(x, a), label=r"$\frac{d^2}{dx^2}\psi_T(x)$")
    plt.plot(x, E_L(x, a), label=r"$E_L(x)$")
    plt.xlabel(r"$x$")
    plt.legend()
    plt.tight_layout()
    plt.savefig("psi_T.png")


def VMC(N_steps, R, d, a, dR=0.1, target_acceptance=0.3, adjust_dR_every=1000):
    """ Variational Monte Carlo """
    N_accepted_total = 0
    N_accepted = 0
    acceptances = []
    for i in range(N_steps):
        # Select a random walker
        w = rng.integers(0, len(R))

        # Shift the walker by a random amount
        R_w_old = R[w].copy()
        R_w_new = R_w_old + rng.uniform(-dR, dR)
        while np.abs(R_w_new) > d / 2:
            R_w_new = R_w_old + rng.uniform(-dR, dR)

        # Calculate the fraction p = [psi_T(R_w_new) / psi_T(R_w_old)]^2
        p = (psi_T(R_w_new, a) / psi_T(R_w_old, a)) ** 2
        if p >= 1 or rng.random() < p:
            R[w] = R_w_new
            N_accepted += 1

        # Adjust the step size
        if i % adjust_dR_every == 0:
            acceptance = N_accepted / adjust_dR_every
            acceptances.append(acceptance)
            if acceptance > target_acceptance:
                dR *= 1.1
            else:
                dR /= 1.1
            N_accepted = 0

    return R, np.mean(acceptances)


def main(make_plots=False):
    if make_plots:
        plots()

    # Set up the simulation
    N = 400  # number of walkers
    d = 10  # system size
    R = rng.random(N) * d - d / 2  # initial positions
    alphas = np.linspace(0.40, 0.60, 5)  # variational parameter
    dR = 0.1  # maximum displacement
    N_steps = 3000 * N

    # Open a file to write the results to
    f = open("results.txt", "w")
    f.write(f"# N = {N}\n")
    f.write(f"# d = {d}\n")
    f.write(f"# dR = {dR}\n")
    f.write(f"# N_steps = {N_steps}\n\n")

    # Run the simulation
    f.write("Running the simulation...\n")
    f.write("# alpha E_mean E_std acceptance\n")
    for alpha in alphas:
        R, acceptance = VMC(N_steps, R, d, alpha, dR=dR)

        # Calculate the energies
        E = np.array([E_L(R[i], alpha) for i in range(N)])
        E_mean = np.mean(E)
        E_std = np.std(E)
        f.write(f"{alpha} {E_mean} {E_std} {acceptance}\n")

        # Plot the results
        plt.hist(R, bins=100, density=True, label="VMC")
        plt.savefig(f"histogram_{alpha}.png")
        plt.close()
    f.close()


if __name__ == "__main__":
    main()
