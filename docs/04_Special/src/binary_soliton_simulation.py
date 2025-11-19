#!/usr/bin/env python3
import numpy as np

def potential(psi, lam=1.0):
    """Simple phi^4-like potential: V = (lam/4) * (psi^2 - 1)^2"""
    return 0.25 * lam * (psi**2 - 1.0)**2

def dV_dpsi(psi, lam=1.0):
    """Derivative of the potential V with respect to psi."""
    return lam * psi * (psi**2 - 1.0)

def relax_single_soliton(nz=2000, zmin=-20.0, zmax=20.0,
                         dz=None, d_tau=1e-3, max_steps=200000, tol=1e-10):
    """Gradient-flow relaxation to construct a single-soliton profile."""
    if dz is None:
        dz = (zmax - zmin) / (nz - 1)
    z = np.linspace(zmin, zmax, nz)
    # Initial guess: localized bump
    psi = np.exp(-0.5 * z**2)
    # Simple Dirichlet boundaries psi = 0 at edges
    for step in range(max_steps):
        lap = (np.roll(psi, -1) - 2*psi + np.roll(psi, 1)) / dz**2
        # enforce boundaries
        lap[0] = lap[-1] = 0.0
        dE_dpsi = -lap + dV_dpsi(psi)
        psi_new = psi - d_tau * dE_dpsi
        psi_new[0] = psi_new[-1] = 0.0
        dE = np.max(np.abs(psi_new - psi))
        psi = psi_new
        if dE < tol:
            break
    return z, psi

def evolve_binary(psi0, z, zA0=5.7, zB0=-4.3,
                  dt=1e-2, nsteps=2000):
    """Time evolution of a binary configuration using a leapfrog scheme."""
    dz = z[1] - z[0]
    # Construct initial field
    def shifted_profile(z, z0):
        return np.interp(z - z0, z, psi0, left=0.0, right=0.0)
    psi = shifted_profile(z, zA0) + shifted_profile(z, zB0)
    psi_prev = np.copy(psi)  # initial time derivative = 0

    traj_A = []
    traj_B = []
    energies = []

    for n in range(nsteps):
        # Spatial laplacian
        lap = (np.roll(psi, -1) - 2*psi + np.roll(psi, 1)) / dz**2
        lap[0] = lap[-1] = 0.0
        # Equation of motion: d2 psi / dt2 = lap - dV/dpsi
        accel = lap - dV_dpsi(psi)
        psi_next = 2*psi - psi_prev + dt**2 * accel
        psi_next[0] = psi_next[-1] = 0.0

        # Simple energy diagnostic (no boundaries term)
        kinetic = ((psi - psi_prev) / dt)**2 / 2.0
        grad = ((np.roll(psi, -1) - psi) / dz)**2 / 2.0
        grad[-1] = 0.0
        pot = potential(psi)
        E = np.trapz(kinetic + grad + pot, z)
        energies.append(E)

        # Track peaks for z_A, z_B (very rough: two largest maxima)
        idx_sorted = np.argsort(psi)[-2:]
        z_peaks = np.sort(z[idx_sorted])
        if len(z_peaks) == 2:
            traj_A.append(z_peaks[1])
            traj_B.append(z_peaks[0])
        else:
            traj_A.append(np.nan)
            traj_B.append(np.nan)

        psi_prev, psi = psi, psi_next

    return np.array(traj_A), np.array(traj_B), np.array(energies)

if __name__ == "__main__":
    z, psi0 = relax_single_soliton()
    zA_traj, zB_traj, E_t = evolve_binary(psi0, z)
    # At this point one can save trajectories and energy to disk
    np.savez("binary_soliton_data.npz", z=z, psi0=psi0,
             zA=zA_traj, zB=zB_traj, E=E_t)
