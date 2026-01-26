import numpy as np
import argparse
import sys

# ============================================================
# Constrained gradient flow for axisymmetric toroidal solitons
# with rigorous Lagrange multiplier λ(t) enforcing Q exactly
# (up to discretization error).
#
# Energy:
#   E = 2π ∫ [ 1/2 |∇ρ|^2 + 1/2 (n^2 ρ^2 / R^2) + U(ρ) ] R dR dz
#
# Constraint:
#   Q = 2π ∫ ρ^2 R dR dz  = Q_target
#
# Gradient flow with constraint:
#   ρ_t = - δE/δρ + λ ρ
# where λ is chosen so that dQ/dt = 0:
#   λ = ( ∫ ρ (δE/δρ) R dR dz ) / ( ∫ ρ^2 R dR dz )
#
# Discretization:
#   Laplacian in conservative form:
#     ∇²ρ = (1/R) d/dR ( R dρ/dR ) + d²ρ/dz²
#   Zero-flux (Neumann) boundaries via mirrored ghost-fluxes.
# ============================================================


# -------------------------
# Domain (cylindrical)
# -------------------------
R_min, R_max = 0.2, 20.0    # keep away from R=0 (punched domain)
Z_max = 20.0
NR, NZ = 400, 400

R = np.linspace(R_min, R_max, NR)
Z = np.linspace(-Z_max, Z_max, NZ)
dR = R[1] - R[0]
dZ = Z[1] - Z[0]

RR, ZZ = np.meshgrid(R, Z, indexing="ij")

# -------------------------
# Physical parameters
# -------------------------
n = 3

# Potential: U(ρ) = 1/2 m^2 ρ^2 - g/3 ρ^3 + h/4 ρ^4
m2 = 1.0
g = 1.0
h = 0.5

# -------------------------
# Numerics
# -------------------------
dt = 5e-5               # try 1e-5..1e-4 (stability depends on grid)
n_steps = 8000          # relaxation steps
report_every = 400

# convergence tolerances
tol_rel_E = 1e-10
tol_force = 1e-7

# stability test settings
stability_steps = 6000
stab_report_every = 300
damping_gamma = 1.0     # for inertial dynamics
perturb_eps = 0.02      # 2% perturbation


# ============================================================
# Utilities: quadratures and operators
# ============================================================

def integrate_axisym(f):
    """Axisymmetric integral: 2π ∫ f(R,z) R dR dz"""
    return 2.0 * np.pi * np.sum(f * RR) * dR * dZ


def compute_Q(rho):
    return integrate_axisym(rho**2)


def U_of_rho(rho):
    return 0.5 * m2 * rho**2 - (g / 3.0) * rho**3 + (h / 4.0) * rho**4


def dU_drho(rho):
    return m2 * rho - g * rho**2 + h * rho**3


def laplacian_axisym_conservative(rho):
    """
    Conservative axisymmetric Laplacian:
      ∇²ρ = (1/R) ∂R( R ∂Rρ ) + ∂zzρ

    Implemented with fluxes:
      (1/R_i) * ( (F_{i+1/2} - F_{i-1/2})/dR )  where
      F_{i+1/2} = R_{i+1/2} * (ρ_{i+1}-ρ_i)/dR

    Neumann BC => zero flux at domain boundaries:
      F_{1/2}=0 at inner boundary, F_{NR-1/2}=0 at outer boundary,
      and similarly in z.
    """
    lap = np.zeros_like(rho)

    # --- Radial part: (1/R) d/dR( R dρ/dR ) ---
    # R half-steps
    R_half_plus  = 0.5 * (R[1:] + R[:-1])  # length NR-1
    # flux F_{i+1/2} for i=0..NR-2
    # shape (NR-1, NZ)
    F_plus = (R_half_plus[:, None] * (rho[1:, :] - rho[:-1, :]) / dR)

    # impose zero flux at boundaries by padding
    # F_{-1/2}=0, F_{NR-1/2}=0
    F_pad = np.zeros((NR + 1, NZ), dtype=rho.dtype)
    # map F_{i+1/2} -> index i+1 in padded (so 1..NR-1)
    F_pad[1:NR, :] = F_plus

    # divergence of flux: (F_{i+1/2}-F_{i-1/2})/dR
    divF = (F_pad[2:, :] - F_pad[:-2, :]) / (2.0 * dR)   # centered
    # divF is shape (NR-1? actually NR-?); adjust by explicit indexing
    # We'll compute with one-sided consistent form instead:
    # Use (F_{i+1/2}-F_{i-1/2})/dR with i=0..NR-1:
    divF = (F_pad[1:NR+1, :] - F_pad[0:NR, :]) / dR      # shape (NR, NZ)

    radial = divF / (R[:, None])

    # --- Axial part: ∂zzρ with zero flux (Neumann) ---
    # flux in z: G_{j+1/2} = (ρ_{j+1}-ρ_j)/dZ
    G_plus = (rho[:, 1:] - rho[:, :-1]) / dZ  # shape (NR, NZ-1)
    G_pad = np.zeros((NR, NZ + 1), dtype=rho.dtype)
    G_pad[:, 1:NZ] = G_plus  # G_{j+1/2} at index j+1
    # zero flux at z boundaries automatically via padding zeros

    divG = (G_pad[:, 1:NZ+1] - G_pad[:, 0:NZ]) / dZ      # shape (NR, NZ)

    lap = radial + divG
    return lap


def grad_components(rho):
    """Finite-difference gradients used only for energy diagnostics."""
    dR_rho = np.zeros_like(rho)
    dZ_rho = np.zeros_like(rho)
    dR_rho[1:-1, :] = (rho[2:, :] - rho[:-2, :]) / (2.0 * dR)
    dZ_rho[:, 1:-1] = (rho[:, 2:] - rho[:, :-2]) / (2.0 * dZ)

    # Neumann edges (mirror)
    dR_rho[0, :]  = 0.0
    dR_rho[-1, :] = 0.0
    dZ_rho[:, 0]  = 0.0
    dZ_rho[:, -1] = 0.0
    return dR_rho, dZ_rho


def compute_energy(rho):
    dR_rho, dZ_rho = grad_components(rho)
    grad2 = dR_rho**2 + dZ_rho**2
    angular = (n**2) * rho**2 / (RR**2)
    dens = 0.5 * grad2 + 0.5 * angular + U_of_rho(rho)
    return integrate_axisym(dens)


def functional_derivative(rho):
    """
    δE/δρ = -∇²ρ + (n^2 / R^2) ρ + dU/dρ
    """
    lap = laplacian_axisym_conservative(rho)
    return -lap + (n**2) * rho / (RR**2) + dU_drho(rho)


def compute_lambda(rho, gE):
    """
    Enforce dQ/dt = 0 for flow ρ_t = -gE + λ ρ
    => λ = (∫ ρ gE R) / (∫ ρ^2 R)
    """
    num = integrate_axisym(rho * gE)
    den = integrate_axisym(rho**2) + 1e-30
    return num / den


# ============================================================
# Initialization helpers
# ============================================================

def gaussian_ring(R0=8.0, sigma_R=1.5, sigma_Z=1.5):
    return np.exp(-((RR - R0)**2 / (2.0 * sigma_R**2) +
                    (ZZ)**2     / (2.0 * sigma_Z**2)))

def normalize_to_Q(rho, Q_target):
    Q0 = compute_Q(rho)
    if Q0 <= 0:
        raise RuntimeError("Initial rho has non-positive Q.")
    return rho * np.sqrt(Q_target / Q0)

def tiny_floor(rho, eps=1e-15):
    # keep strictly nonnegative while avoiding exactly-zero divisions
    return np.maximum(rho, eps)


# ============================================================
# Core: constrained relaxation
# ============================================================

def relax_constrained_aggressive(Q_target, rho0=None,
                                 dt_init=1e-4,
                                 dt_min=1e-7,
                                 dt_max=5e-3,
                                 max_steps=5000000,
                                 report_every=500,
                                 tol_rel_E=1e-11,
                                 tol_force=5e-8):
    """
    Constrained gradient flow con line search sul passo temporale.

    ρ_{t} = - δE/δρ + λ(t) ρ
    con λ scelto da dQ/dt=0, e dt adattivo:

      - tenta uno step con dt;
      - se E_new > E_old: rifiuta step, dt <- dt/2;
      - se E_new << E_old e nessuna instabilità: ogni tanto dt <- dt * growth;

    Stop quando:
      - relΔE << tol_rel_E
      - ||F||_weighted << tol_force
    """

    if rho0 is None:
        rho = gaussian_ring()
    else:
        rho = rho0.copy()

    rho = tiny_floor(rho)
    rho = normalize_to_Q(rho, Q_target)

    dt = dt_init
    E_prev = compute_energy(rho)

    print(f"\n=== Aggressive relaxation to Q = {Q_target:.6g} ===")
    print(f"Initial: E={E_prev:.12e}, Q={compute_Q(rho):.12e}, dt={dt:.3e}")

    accepted_steps = 0

    for k in range(1, max_steps + 1):
        # calcolo derivata funzionale
        gE = functional_derivative(rho)
        lam = compute_lambda(rho, gE)
        F = -gE + lam * rho

        # norma forza (per diagnosi e stop)
        F2 = integrate_axisym(F**2)
        rho2 = integrate_axisym(rho**2)
        fnorm = np.sqrt(F2 / (rho2 + 1e-30))

        # prova uno step esplicito con dt attuale
        attempt_ok = False
        n_backtrack = 0

        while not attempt_ok:
            if dt < dt_min:
                print("  !! dt è sceso sotto dt_min, mi fermo (stiffness).")
                return rho

            rho_trial = rho + dt * F
            rho_trial = tiny_floor(rho_trial)
            rho_trial = normalize_to_Q(rho_trial, Q_target)

            E_trial = compute_energy(rho_trial)

            if E_trial <= E_prev:  # energia non cresce: accetto
                attempt_ok = True
            else:
                # energia sale ⇒ backtracking
                dt *= 0.5
                n_backtrack += 1
                if n_backtrack > 20:
                    print("  !! Troppi backtracking, qualcosa è rigido/instabile.")
                    return rho

        # step accettato
        rho = rho_trial
        E_now = E_trial
        accepted_steps += 1

        relE = abs(E_now - E_prev) / (abs(E_prev) + 1e-30)
        E_prev = E_now

        # piccola strategia di aumento dt, solo se si va lisci
        if (accepted_steps % 50) == 0 and n_backtrack == 0:
            dt = min(dt * 1.2, dt_max)

        # log periodico
        if (k % report_every) == 0 or k == 1:
            Q_now = compute_Q(rho)
            print(f" step={k:6d}  E={E_now:.12e}  Q={Q_now:.12e}  "
                  f"λ={lam:.6e}  dt={dt:.3e}  relΔE={relE:.3e}  ||F||={fnorm:.3e}")

        # criteri di convergenza stretti
        if relE < tol_rel_E and fnorm < tol_force:
            Q_now = compute_Q(rho)
            print(f" Converged (aggressive): step={k}, "
                  f"E={E_now:.12e}, Q={Q_now:.12e}, dt={dt:.3e}, ||F||={fnorm:.3e}")
            return rho

    print(" Warning: reached max_steps in aggressive mode without full convergence.")
    return rho


# ============================================================
# Stability / “tenuta nel tempo” test
# ============================================================

def stability_test(rho_star, Q_target, seed=0):
    """
    Inertial damped dynamics with constraint:
      ρ_tt + γ ρ_t = - δE/δρ + λ ρ
    Use velocity v = ρ_t, time-stepped with semi-explicit scheme.

    If stable, the perturbed state should relax back:
      energy decreases and key norms remain bounded,
      without runaway concentration or drift to boundary.

    Reports:
      - E(t), Q(t)
      - max(rho), min(rho)
      - a "compactness" proxy via second moment <(R-Rc)^2 + z^2> (weighted)
    """
    rng = np.random.default_rng(seed)

    rho = rho_star.copy()
    rho = tiny_floor(rho)
    rho = normalize_to_Q(rho, Q_target)

    # Perturbation: multiplicative low-amplitude noise, smoothed-ish
    noise = rng.normal(size=rho.shape)
    noise = noise / (np.std(noise) + 1e-30)
    rho = rho * (1.0 + perturb_eps * noise)
    rho = tiny_floor(rho)
    rho = normalize_to_Q(rho, Q_target)

    v = np.zeros_like(rho)

    def compactness_metrics(rho):
        w = rho**2
        W = integrate_axisym(w) + 1e-30
        Rc = integrate_axisym(RR * w) / W
        Zc = integrate_axisym(ZZ * w) / W
        var = integrate_axisym(((RR - Rc)**2 + (ZZ - Zc)**2) * w) / W
        return Rc, Zc, np.sqrt(var)

    E0 = compute_energy(rho)
    Rc0, Zc0, sigma0 = compactness_metrics(rho)

    print("\n=== Stability test (damped inertial dynamics) ===")
    print(f"Perturb eps={perturb_eps:.3g}, gamma={damping_gamma:.3g}")
    print(f"t=0: E={E0:.12e}, Q={compute_Q(rho):.12e}, "
          f"maxρ={rho.max():.6e}, σ={sigma0:.6e}, Rc={Rc0:.6e}")

    # thresholds for "collapse" heuristics (tunable)
    maxrho_blow = 50.0 * rho_star.max()
    sigma_shrink = 0.25 * sigma0

    for k in range(1, stability_steps + 1):
        gE = functional_derivative(rho)
        lam = compute_lambda(rho, gE)

        # acceleration: a = -γ v + (-gE + λ ρ)
        a = -damping_gamma * v + (-gE + lam * rho)

        # time step (semi-explicit)
        v = v + dt * a
        rho = rho + dt * v

        rho = tiny_floor(rho)
        # keep Q fixed (tiny correction)
        rho = normalize_to_Q(rho, Q_target)

        if (k % stab_report_every) == 0 or k == 1:
            E = compute_energy(rho)
            Q = compute_Q(rho)
            Rc, Zc, sigma = compactness_metrics(rho)
            mx = rho.max()
            mn = rho.min()

            print(f" step={k:6d}  E={E:.12e}  Q={Q:.12e}  "
                  f"λ={lam:.6e}  maxρ={mx:.6e}  minρ={mn:.3e}  "
                  f"σ={sigma:.6e}  Rc={Rc:.6e}")

            # collapse / runaway heuristics
            if mx > maxrho_blow and sigma < sigma_shrink:
                print(" >>> Likely COLLAPSE / runaway concentration detected.")
                return False

            # drift to boundary (Rc near R_max or R_min)
            if Rc > (R_max - 2.0*dR) or Rc < (R_min + 2.0*dR):
                print(" >>> Likely DRIFT to domain boundary (box not large enough or unstable).")
                return False

    # If we got here: bounded and relaxed-ish
    print(" >>> No runaway detected in this horizon. Likely stable (within this model & box).")
    return True


# ============================================================
# Main (example single Q)
# ============================================================

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Esegue il rilassamento con un Q_target.")
    
    parser.add_argument(
        'Q_valore',         # Nome che useremo per recuperare l'argomento
        type=float,         # Forza la conversione a float (es. 1800 -> 1800.0)
        help='Il valore target Q per il rilassamento.'
    )
    
    args = parser.parse_args()
    
    Q_target = args.Q_valore

    # primo run (puoi passare rho0=None o un seed precedente)
    rho_star = relax_constrained_aggressive(Q_target, rho0=None)

    E_star = compute_energy(rho_star)
    print(f"\nStationary candidate (aggressive): "
          f"E={E_star:.12e}, Q={compute_Q(rho_star):.12e}, maxρ={rho_star.max():.6e}")

    ok = stability_test(rho_star, Q_target, seed=1)
    print("\nSTABILITY:", "PASS" if ok else "FAIL")

