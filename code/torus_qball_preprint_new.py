import numpy as np
import matplotlib.pyplot as plt

# ================================================================
# Toroidal Q-ball / organized motion: rigorous gradient-flow solver
# ================================================================

# Domain parameters (cylindrical coordinates)
R_min, R_max = 0.2, 20.0   # avoid the R=0 axis to remove the 1/R^2 singularity
Z_max = 20.0
NR, NZ = 400, 400

R = np.linspace(R_min, R_max, NR)
Z = np.linspace(-Z_max, Z_max, NZ)
dR = R[1] - R[0]
dZ = Z[1] - Z[0]
RR, ZZ = np.meshgrid(R, Z, indexing="ij")

# Winding number (phase ~ e^{i n phi})
n = 3

# Potential parameters: U(rho) = 1/2 m^2 rho^2 - g/3 rho^3 + h/4 rho^4
m2 = 1.0
g = 1.0
h = 0.5

# Target charge
Q_target = 200.0

# Initial torus profile (Gaussian ring)
R0 = 8.0
sigma_R = 1.5
sigma_Z = 1.5

rho = np.exp(-((RR - R0)**2 / (2.0 * sigma_R**2) +
               ZZ**2        / (2.0 * sigma_Z**2)))

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
    Discretized with centered finite differences and Neumann BC at the box.
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

    # Neumann-like boundary conditions (zero normal derivative)
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
    Energy density corresponding to the functional:
        E[rho] = 2*pi ∫ [ 1/2 |∇rho|^2
                          + 1/2 (n^2 rho^2 / R^2)
                          + U(rho) ] R dR dz
    This choice makes δE/δrho = -∇^2 rho + (n^2 rho / R^2) + dU/drho.
    """
    drho_dR = np.zeros_like(rho)
    drho_dZ = np.zeros_like(rho)
    drho_dR[1:-1, :] = (rho[2:, :] - rho[:-2, :]) / (2.0 * dR)
    drho_dZ[:, 1:-1] = (rho[:, 2:] - rho[:, :-2]) / (2.0 * dZ)
    grad2 = drho_dR**2 + drho_dZ**2

    angular = (n**2) * rho**2 / (RR**2)

    U = 0.5 * m2 * rho**2 - (g / 3.0) * rho**3 + (h / 4.0) * rho**4

    # Note the factor 1/2 in front of both gradient and angular parts.
    return 0.5 * grad2 + 0.5 * angular + U

def compute_energy(rho):
    return 2.0 * np.pi * np.sum(energy_density(rho) * RR) * dR * dZ

# Normalize initial Q exactly to Q_target
Q0 = compute_Q(rho)
rho *= np.sqrt(Q_target / (Q0 + 1e-16))
print(f"Initial Q normalized to: {compute_Q(rho):.6f}")

# Gradient flow parameters
dt = 1e-4
n_steps = 4000
sample_every = 200

E_hist = []
Q_hist = []
steps = []

for step in range(n_steps + 1):
    if step % sample_every == 0:
        E_hist.append(compute_energy(rho))
        Q_hist.append(compute_Q(rho))
        steps.append(step)
        print(f"step = {step:5d}, E = {E_hist[-1]:.8e}, Q = {Q_hist[-1]:.6f}")

    lap = laplacian_cylindrical(rho)

    # Functional derivative δE/δrho for the chosen E[rho]
    force = (
        -lap
        + (n**2) * rho / (RR**2)
        + dU_drho(rho)
    )

    # Gradient flow step: ∂_t rho = - δE/δrho
    rho -= dt * force

    # Enforce positivity of rho (rho is a modulus)
    rho = np.maximum(rho, 0.0)

    # Project back to Q = Q_target (charge-constrained minimization)
    Q_now = compute_Q(rho)
    rho *= np.sqrt(Q_target / (Q_now + 1e-16))

# Measure torus radii from rho(R,0) using FWHM
iz0 = np.argmin(np.abs(Z))
rho_R = rho[:, iz0]
rho_max = rho_R.max()
thresh = rho_max / 2.0
indices = np.where(rho_R > thresh)[0]

if len(indices) >= 2:
    R_in = R[indices[0]]
    R_out = R[indices[-1]]
    R_center = 0.5 * (R_in + R_out)
    a = 0.5 * (R_out - R_in)
    print("\n=== Toroidal radii (half-maximum) ===")
    print(f"rho_max = {rho_max:.6f}")
    print(f"R_in    = {R_in:.3f}")
    print(f"R_out   = {R_out:.3f}")
    print(f"R_c     = {R_center:.3f}")
    print(f"a       = {a:.3f}")
else:
    print("Could not determine R_in and R_out (no clear FWHM region).")

# === Plots ===

# 1) rho(R,z) heatmap (meridional section)
plt.figure(figsize=(8, 5))
plt.imshow(
    rho.T,
    origin="lower",
    extent=[R_min, R_max, -Z_max, Z_max],
    aspect="auto",
    cmap="viridis"
)
plt.colorbar(label="rho")
plt.xlabel("R")
plt.ylabel("z")
plt.title(f"rho(R,z) for Q={Q_target:g}, n={n}")
plt.tight_layout()
plt.savefig("figures/rho_RZ_Q200_rigorous.png", dpi=200)

# 2) radial profile rho(R,0)
plt.figure(figsize=(7, 5))
plt.plot(R, rho_R, label=r"$\rho(R,0)$")
plt.axhline(thresh, linestyle="--", label="FWHM level")
plt.xlabel("R")
plt.ylabel("rho(R,0)")
plt.title(f"Radial profile at z=0, Q={Q_target:g}, n={n}")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("figures/rho_profile_Q200_rigorous.png", dpi=200)

# 3) reconstruction in x-y plane at z=0
Lxy = 15.0
Nx = Ny = 400
x = np.linspace(-Lxy, Lxy, Nx)
y = np.linspace(-Lxy, Lxy, Ny)
XX, YY = np.meshgrid(x, y, indexing="ij")
R_xy = np.sqrt(XX**2 + YY**2)
rho_xy = np.interp(R_xy, R, rho_R, left=0.0, right=0.0)

plt.figure(figsize=(6, 6))
plt.imshow(
    rho_xy.T,
    origin="lower",
    extent=[-Lxy, Lxy, -Lxy, Lxy],
    aspect="equal",
    cmap="viridis"
)
plt.colorbar(label="rho")
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"rho(x,y,z=0) for Q={Q_target:g}, n={n}")
plt.tight_layout()
plt.savefig("figures/rho_xy_Q200_rigorous.png", dpi=200)

# 4) energy vs steps
plt.figure(figsize=(7, 5))
plt.plot(steps, E_hist, marker="o")
plt.xlabel("step")
plt.ylabel("E")
plt.title(f"Energy vs gradient flow steps (Q={Q_target:g}, n={n})")
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/energy_flow_Q200_rigorous.png", dpi=200)

plt.close("all")

