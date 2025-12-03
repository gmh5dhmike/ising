import numpy as np
import time
import os
import math

import matplotlib
matplotlib.use("Agg")   # no display on Rivanna

# Lattice size and parameters
NX = 64
NY = 64
Nspins = NX * NY
ntherm = 1000           # thermalization sweeps before measuring
VisualDisplay = 0       # set to 1 if you want ASCII display
SleepTime = 200000      # in microseconds for display mode


def initialize_hot():
    """Random initial configuration of spins (+1/-1) on an NX x NY lattice."""
    spins = np.where(np.random.rand(NX, NY) < 0.5, 1, -1).astype(np.int8)
    return spins


def sweep(spins, beta, h):
    """
    One Metropolis sweep: visit every site once and try to flip.
    Periodic boundary conditions. J = 1, k_B = 1.
    h is the dimensionless field H/(k_B T). For the assignment use h = 0.
    """
    for i in range(NX):
        ip = (i + 1) % NX   # right
        im = (i - 1) % NX   # left
        for j in range(NY):
            jp = (j + 1) % NY   # up
            jm = (j - 1) % NY   # down

            s = spins[i, j]
            nb = spins[ip, j] + spins[im, j] + spins[i, jp] + spins[i, jm]

            # Physical energy change: ΔE = 2 * s * (J * nb + H),
            # with J = 1, and H = h * T (because h = H / (k_B T)).
            # But for h = 0 this reduces to ΔE = 2 s nb.
            # We'll implement the general ΔE for correctness:
            # H = h * T, but we already have beta = 1/T, so H = h / beta.
            H = h / beta
            dE = 2.0 * s * (nb + H)

            # Metropolis acceptance with probability exp(-beta * ΔE)
            if dE <= 0.0 or np.random.rand() < math.exp(-beta * dE):
                spins[i, j] = -s

    return spins


def energy(spins, h, T):
    """
    Total physical energy E of the 2D Ising config with J = 1 and field H = h*T.
    Periodic BC.
    E = - sum_{<ij>} s_i s_j - H sum_i s_i
    """
    H = h * T  # because h = H / (k_B T) and k_B = 1
    E = 0.0

    for i in range(NX):
        ip = (i + 1) % NX
        for j in range(NY):
            jp = (j + 1) % NY
            s = spins[i, j]
            # count bonds to the right and up only (to avoid double-counting)
            E -= s * (spins[ip, j] + spins[i, jp])

    # field term
    M = spins.sum()
    E -= H * M

    return E


def display_lattice(T, spins):
    """Optional ASCII display of lattice and magnetization."""
    if SleepTime > 0:
        os.system('cls' if os.name == 'nt' else 'clear')

    chars = np.where(spins == 1, 'X', '-')
    for row in chars:
        print(''.join(row))

    m = spins.mean()
    print(f"T = {T:.6f}   magnetization <sigma> = {m:.6f}")

    if SleepTime > 0:
        time.sleep(SleepTime / 1_000_000.0)
    else:
        print()


def main():
    output_filename = "ising2d_vs_T_EMC.dat"

    print(f"Program calculates M, E, and C vs. T for a 2D Ising model of "
          f"{NX}x{NY} spins with periodic boundary conditions.\n")

    np.random.seed(int(time.time()))

    nsweep = int(input("Enter # sweeps per temperature sample:\n"))
    h = float(input("Enter value of magnetic field parameter h (use 0.0 for this assignment):\n"))
    Tmax = float(input("Enter starting value (maximum) of temperature T (=1/beta):\n"))
    ntemp = int(input("Enter # temperatures to simulate:\n"))

    spins = initialize_hot()

    with open(output_filename, 'w') as output:
        # simulate temperatures from Tmax down to ~0
        for itemp in range(ntemp, 0, -1):
            T = (Tmax * itemp) / ntemp
            beta = 1.0 / T

            print(f"Simulating T = {T:.3f} ...")
            # --- thermalization ---
            for _ in range(ntherm):
                spins = sweep(spins, beta, h)

            # --- measurement ---
            sumM = 0.0
            sumE = 0.0
            sumE2 = 0.0

            for _ in range(nsweep):
                spins = sweep(spins, beta, h)

                E = energy(spins, h, T)
                M = spins.sum()

                sumM += M
                sumE += E
                sumE2 += E * E

            # averages
            M_mean_total = sumM / nsweep   # total magnetization
            E_mean_total = sumE / nsweep   # total energy
            E2_mean_total = sumE2 / nsweep

            m = M_mean_total / Nspins      # magnetization per spin
            e = E_mean_total / Nspins      # energy per spin

            # specific heat per spin:
            # C = ( <E^2> - <E>^2 ) / (N k_B T^2), with k_B = 1
            C = (E2_mean_total - E_mean_total**2) / (Nspins * T * T)

            output.write(f"{T:.6f} {m:.6f} {e:.6f} {C:.6f}\n")

            if VisualDisplay:
                display_lattice(T, spins)

    print(f"\nDone. Output file is {output_filename}")


if __name__ == "__main__":
    main()

