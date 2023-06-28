""" Lennard-Jones potential """
import matplotlib.pyplot as plt
import numpy as np


def potential(r, epsilon=0.010323, sigma=3.405, r_tail=7, r_cut=7.5, A=-6.8102128E-3, B=-5.5640876E-3):
    """ Lennard-Jones potential """
    if r > r_cut:
        return 0
    elif r > r_tail:
        return A * (r - r_cut) ** 3 + B * (r - r_cut) ** 2
    else:
        return 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)


def force(r, epsilon=0.010323, sigma=3.405, r_tail=7, r_cut=7.5, A=-6.8102128E-3, B=-5.5640876E-3):
    """ Lennard-Jones force """
    if r > r_cut:
        return 0
    elif r > r_tail:
        return 3 * A * (r - r_cut) ** 2 + 2 * B * (r - r_cut)
    else:
        return 24 * epsilon * (-2 * (sigma / r) ** 12 + (sigma / r) ** 6) / r


def plot_potential(sigma=3.405, r_tail=7, r_cut=7.5):
    r = np.linspace(3.3, 8, 1000)
    plt.plot(r, [potential(r_) for r_ in r])
    plt.xlabel("r")
    plt.ylabel("V(r)")
    plt.axhline(y=0, color="black", linestyle="--")
    plt.axvline(x=sigma, color="black", linestyle="--")
    r_min = 2 ** (1 / 6) * sigma
    plt.axvline(x=r_min, color="black", linestyle="--")
    plt.axvline(x=r_tail, color="black", linestyle="--")
    plt.axvline(x=r_cut, color="black", linestyle="--")
    plt.text(sigma + 0.1, 0.01, "σ")
    plt.text(r_tail + 0.1, 0.01, "$r_{tail}$")
    plt.text(r_cut + 0.1, 0.01, "$r_{cut}$")
    plt.tight_layout()
    plt.savefig("lennard_jones.png")
    plt.close()


def plot_force(sigma=3.405, r_tail=7, r_cut=7.5):
    r = np.linspace(3.3, 8, 1000)
    plt.plot(r, [force(r_) for r_ in r])
    plt.xlabel("r")
    plt.ylabel("F(r)")
    plt.axhline(y=0, color="black", linestyle="--")
    plt.axvline(x=sigma, color="black", linestyle="--")
    r_min = 2 ** (1 / 6) * sigma
    plt.axvline(x=r_min, color="black", linestyle="--")
    plt.axvline(x=r_tail, color="black", linestyle="--")
    plt.axvline(x=r_cut, color="black", linestyle="--")
    plt.text(sigma + 0.1, 0.005, "σ")
    plt.text(r_tail + 0.1, 0.005, "$r_{tail}$")
    plt.text(r_cut + 0.1, 0.005, "$r_{cut}$")
    plt.tight_layout()
    plt.savefig("lennard_jones_force.png")
    plt.close()


def main(makes_plots=False):
    if makes_plots:
        plot_potential()
        plot_force()


if __name__ == "__main__":
    main()
