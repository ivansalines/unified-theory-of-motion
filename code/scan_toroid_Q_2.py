import numpy as np

# ================================================================
# Scan of toroidal solitons vs charge Q
# Using the rigorous gradient-flow solver
# ================================================================

# ---------- Domain and physical parameters ----------

R_min, R_max = 0.2, 20.0   # avoid R=0 singularity
Z_max = 20.0
NR, NZ = 400, 400

R = np.linspace(R_min, R_max, NR)
Z = np.linspace(-Z_max, Z_max, NZ)
dR = R[1] - R[0]
dZ = Z[1] - Z[0]
RR, ZZ = np.meshgrid(R, Z, indexing="ij")

# Winding number
n = 2

# Potential parameters: U(rho) = 1/2 m^2 rho^2 - g/3 rho^3 + h/4 rho^4
m2 = 1.0
g = 1.0
h = 0.5

# Gradient flow parameters
dt = 1e-4
n_steps = 4000
sample_every = 500

# List of charges to explore
Q_values = [50.0, 80.0, 120.0, 200.0, 300.0, 500.0, 800.0, 1200.0]


# ---------- Core functions ----------

def compute_Q(rho):
    """
    Charge functional:
        Q = 2*pi ∫ rho^2(R,z) * R dR dz
    """
    return 2.0 * np.pi * np.sum(rho**2 * RR) * dR * dZ


def laplacian_cylindrical(rho):
    """
    Cylindrical Laplacian with axial symmetry:
        ∇^2 rho = d^2 rho/dR^2 + (1/R) d rho/dR + d^2 rho/dz^2
    Neumann-like BCs at box boundaries.
    """
    lap = np.zeros_like(rho)

    # second derivatives in R and z (interior points)
    lap[1:-1, 1:-1] = (
        (rho[2:, 1:-1] - 2.0 * rho[1:-1, 1:-1] + rho[:-2, 1:-1]) / dR**2 +
        (rho[1:-1, 2:] - 2.0 * rho[1:-1, 1:-1] + rho[1:-1, :-2]) / dZ**2
    )

    # first derivative in R for (1/R) d rho/dR term
    drho_dR = np.zeros_like(rho)
    drho_dR[1:-1, :] = (rho[2:, :] - rho[:-2, :]) / (2.0 * dR)
    lap[1:-1, 1:-1] += drho_dR[1:-1, 1:-1] / RR[1:-1, 1:-1]

    # Neumann-like boundaries
    lap[0, :]  = lap[1, :]
    lap[-1, :] = lap[-2, :]
    lap[:, 0]  = lap[:, 1]
    lap[:, -1] = lap[:, -2]

    return lap


def dU_drho(rho):
    """
    dU/drho for:
        U(rho) = 1/2 m^2 rho^2 - g/3 rho^3 + h/4 rho^4
    """
    return m2 * rho - g * rho**2 + h * rho**3


def energy_density(rho):
    """
    Energy density corresponding to:
        E[rho] = 2*pi ∫ [ 1/2 |∇rho|^2
                          + 1/2 (n^2 rho^2 / R^2)
                          + U(rho) ] R dR dz
    """
    drho_dR = np.zeros_like(rho)
    drho_dZ = np.zeros_like(rho)
    drho_dR[1:-1, :] = (rho[2:, :] - rho[:-2, :]) / (2.0 * dR)
    drho_dZ[:, 1:-1] = (rho[:, 2:] - rho[:, :-2]) / (2.0 * dZ)
    grad2 = drho_dR**2 + drho_dZ**2

    angular = (n**2) * rho**2 / (RR**2)
    U = 0.5 * m2 * rho**2 - (g / 3.0) * rho**3 + (h / 4.0) * rho**4

    return 0.5 * grad2 + 0.5 * angular + U


def compute_energy(rho):
    return 2.0 * np.pi * np.sum(energy_density(rho) * RR) * dR * dZ


def measure_torus_radii(rho):
    """
    Extract R_c and a from rho(R,z=0) using FWHM.
    Returns (R_c, a) or (None, None) if FWHM region not found.
    """
    iz0 = np.argmin(np.abs(Z))
    rho_R = rho[:, iz0]
    rho_max = rho_R.max()
    if rho_max <= 0:
        return None, None

    thresh = rho_max / 2.0
    indices = np.where(rho_R > thresh)[0]

    if len(indices) < 2:
        return None, None

    R_in = R[indices[0]]
    R_out = R[indices[-1]]
    R_c = 0.5 * (R_in + R_out)
    a = 0.5 * (R_out - R_in)
    return R_c, a


def gaussian_ring_profile():
    """
    Default initial torus: Gaussian ring centered at R0 with widths sigma_R, sigma_Z.
    """
    R0 = 8.0
    sigma_R = 1.5
    sigma_Z = 1.5
    return np.exp(-((RR - R0)**2 / (2.0 * sigma_R**2) +
                    ZZ**2        / (2.0 * sigma_Z**2)))


def relax_torus(Q_target, rho_seed=None):
    """
    Run gradient flow at fixed Q_target starting from:
      - rho_seed (if provided, rescaled),
      - otherwise a Gaussian ring profile.
    Returns:
      rho_final, E_final, R_c, a
    """
    if rho_seed is None:
        rho = gaussian_ring_profile()
    else:
        rho = rho_seed.copy()

    # Normalize initial Q
    Q0 = compute_Q(rho)
    if Q0 <= 0:
        raise RuntimeError("Initial rho has zero or negative charge.")
    rho *= np.sqrt(Q_target / (Q0 + 1e-16))

    print(f"\n=== Relaxing torus for Q_target = {Q_target:.1f} ===")
    print(f"Initial Q normalized to: {compute_Q(rho):.6f}")

    for step in range(n_steps + 1):
        if step % sample_every == 0:
            E_now = compute_energy(rho)
            Q_now = compute_Q(rho)
            print(f"  step = {step:5d}, E = {E_now:.8e}, Q = {Q_now:.6f}")

        lap = laplacian_cylindrical(rho)

        # Functional derivative δE/δrho
        force = (
            -lap
            + (n**2) * rho / (RR**2)
            + dU_drho(rho)
        )

        # Gradient flow step
        rho -= dt * force

        # Enforce positivity
        rho = np.maximum(rho, 0.0)

        # Project back to Q = Q_target
        Q_now = compute_Q(rho)
        rho *= np.sqrt(Q_target / (Q_now + 1e-16))

    # Final measurements
    E_final = compute_energy(rho)
    R_c, a = measure_torus_radii(rho)
    return rho, E_final, R_c, a


# ---------- Scan over Q and collect data ----------

results = []  # each entry: (Q, E, R_c, a, shape_ratio, E_over_Q)

rho_seed = None

for Q_target in Q_values:
    rho_seed, E_final, R_c, a = relax_torus(Q_target, rho_seed)

    if R_c is not None and a is not None and a > 0:
        shape_ratio = R_c / a
    else:
        shape_ratio = np.nan

    E_over_Q = E_final / Q_target

    results.append((Q_target, E_final, R_c, a, shape_ratio, E_over_Q))

# ---------- Print final table ----------

print("\n=== Summary over Q ===")
print("{:>8s} {:>14s} {:>10s} {:>10s} {:>12s} {:>12s}".format(
    "Q", "E(Q)", "R_c", "a", "R_c/a", "E/Q"
))
for Q, E, Rc, a, chi, e_over_q in results:
    print("{:8.1f} {:14.6e} {:10.4f} {:10.4f} {:12.4f} {:12.6f}".format(
        Q, E,
        Rc if Rc is not None else float('nan'),
        a if a is not None else float('nan'),
        chi,
        e_over_q
    ))

