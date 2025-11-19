import numpy as np
import matplotlib.pyplot as plt

# Shared grid and parameters (same as main simulation)
R_max, Z_max = 20.0, 20.0
NR, NZ = 400, 400

R = np.linspace(0.0, R_max, NR)
Z = np.linspace(-Z_max, Z_max, NZ)
dR = R[1] - R[0]
dZ = Z[1] - Z[0]
RR, ZZ = np.meshgrid(R, Z, indexing="ij")

n = 3
m2 = 1.0
g = 1.0
h = 0.5
alpha_core = 20.0
R_core = 1.0

R0 = 8.0
sigma_R = 1.5
sigma_Z = 1.5

dt = 1e-4
n_steps = 4000
sample_every = 400

def init_rho_ring():
    return np.exp(-((RR - R0)**2 / (2.0 * sigma_R**2) +
                    ZZ**2        / (2.0 * sigma_Z**2)))

def compute_Q(rho):
    return 2.0 * np.pi * np.sum(rho**2 * RR) * dR * dZ

def laplacian_cylindrical(rho):
    lap = np.zeros_like(rho)
    lap[1:-1, 1:-1] = (
        (rho[2:, 1:-1] - 2.0 * rho[1:-1, 1:-1] + rho[:-2, 1:-1]) / dR**2 +
        (rho[1:-1, 2:] - 2.0 * rho[1:-1, 1:-1] + rho[1:-1, :-2]) / dZ**2
    )
    drho_dR = np.zeros_like(rho)
    drho_dR[1:-1, :] = (rho[2:, :] - rho[:-2, :]) / (2.0 * dR)
    lap[1:-1, 1:-1] += drho_dR[1:-1, 1:-1] / (RR[1:-1, 1:-1] + 1e-8)
    lap[0, :]  = lap[1, :]
    lap[-1, :] = lap[-2, :]
    lap[:, 0]  = lap[:, 1]
    lap[:, -1] = lap[:, -2]
    return lap

def dU_drho(rho):
    return m2 * rho - g * rho**2 + h * rho**3

def energy_density(rho):
    drho_dR = np.zeros_like(rho)
    drho_dZ = np.zeros_like(rho)
    drho_dR[1:-1, :] = (rho[2:, :] - rho[:-2, :]) / (2.0 * dR)
    drho_dZ[:, 1:-1] = (rho[:, 2:] - rho[:, :-2]) / (2.0 * dZ)
    grad2 = drho_dR**2 + drho_dZ**2
    angular = (n**2) * rho**2 / (RR**2 + 1e-8)
    U = 0.5 * m2 * rho**2 - (g / 3.0) * rho**3 + (h / 4.0) * rho**4
    core = alpha_core * np.exp(-(RR / R_core)**2) * rho**2
    return grad2 + angular + U + core

def compute_energy(rho):
    return 2.0 * np.pi * np.sum(energy_density(rho) * RR) * dR * dZ

def measure_torus(rho):
    iz0 = np.argmin(np.abs(Z))
    rho_R = rho[:, iz0]
    rho_max = rho_R.max()
    thresh = rho_max / 2.0
    indices = np.where(rho_R > thresh)[0]
    if len(indices) < 2:
        return {
            "rho_max": rho_max,
            "thresh": thresh,
            "R_in": np.nan,
            "R_out": np.nan,
            "R_c": np.nan,
            "a": np.nan,
        }
    R_in = R[indices[0]]
    R_out = R[indices[-1]]
    R_c = 0.5 * (R_in + R_out)
    a = 0.5 * (R_out - R_in)
    return {
        "rho_max": rho_max,
        "thresh": thresh,
        "R_in": R_in,
        "R_out": R_out,
        "R_c": R_c,
        "a": a,
    }

def build_torus_for_Q(Q_target):
    rho = init_rho_ring()
    Q0 = compute_Q(rho)
    rho *= np.sqrt(Q_target / (Q0 + 1e-16))
    print(f"\n=== Q_target = {Q_target} ===")
    print(f"Initial Q normalized: {compute_Q(rho):.6f}")
    for step in range(n_steps + 1):
        if step % sample_every == 0:
            E_now = compute_energy(rho)
            Q_now = compute_Q(rho)
            print(f" step = {step:5d}  E = {E_now:.6e}  Q = {Q_now:.6f}")
        lap = laplacian_cylindrical(rho)
        force = (
            -lap
            + (n**2) * rho / (RR**2 + 1e-8)
            + dU_drho(rho)
            + 2.0 * alpha_core * np.exp(-(RR / R_core)**2) * rho
        )
        rho -= dt * force
        rho = np.maximum(rho, 0.0)
        Q_now = compute_Q(rho)
        rho *= np.sqrt(Q_target / (Q_now + 1e-16))
    E_final = compute_energy(rho)
    meas = measure_torus(rho)
    print(" -> Final energy =", E_final)
    print(" -> Torus measure:", meas)
    return rho, E_final, meas

Q_list = [100.0, 200.0, 400.0]
results = []

for Q_target in Q_list:
    rho_final, E_final, meas = build_torus_for_Q(Q_target)
    results.append({"Q": Q_target, "E": E_final, **meas})

print("\n====================== FINAL RESULTS ======================")
print(" Q_target    E_final      E/Q       R_in      R_out      R_c        a")
print("---------------------------------------------------------------------")
for res in results:
    EQ = res["E"] / res["Q"]
    print(f" {res['Q']:7.1f}  {res['E']:11.4e}  {EQ:7.3f}  "
          f"{res['R_in']:8.3f}  {res['R_out']:8.3f}  "
          f"{res['R_c']:8.3f}  {res['a']:8.3f}")

# Save normalized radial profiles for figure
plt.figure(figsize=(7, 5))
for res, Q_target in zip(results, Q_list):
    rho_final, _, _ = build_torus_for_Q(Q_target)
    iz0 = np.argmin(np.abs(Z))
    rho_R = rho_final[:, iz0]
    rho_R_norm = rho_R / rho_R.max()
    plt.plot(R, rho_R_norm, label=f"Q={Q_target:g}")
plt.xlabel("R")
plt.ylabel("rho(R,0) / max")
plt.title("Normalized radial profiles for different Q")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/rho_profiles_Q100_200_400.png", dpi=200)
plt.close()
