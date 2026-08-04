# Mechanical Whip Tangent Milestone

Archive date: 2026-08-04

This archive preserves the numerical evidence chain that led from the local Hopf threshold and support-feedback diagnostics to the identification of a recurrent, antiperiodic tangent-line stroke on the true rotating clock.

The archive is intentionally organized as a frozen milestone. It contains the uploaded reports, tables, raw arrays, logs, plots, one available analysis script, English-only interpretive notes, and SHA-256 integrity checks.

## Central result

On the unwrapped support-rotation coordinate

\[
s = \Theta/(2\pi),
\]

the leading tangent direction satisfies, to numerical precision,

\[
w(s+P_{\mathrm{line}}) \approx -w(s),
\]

and

\[
w(s+2P_{\mathrm{line}}) \approx w(s).
\]

For the two retained regimes:

| Q | Tangent-line period | Oriented-vector period | Signed overlap at line period | Signed overlap at oriented period |
|---:|---:|---:|---:|---:|
| 522.25 | 0.223611111111 rotations | 0.448611111111 rotations | -0.999985788531 | +0.999958790584 |
| 550 | 0.227777777778 rotations | 0.454166666667 rotations | -0.999978945699 | +0.999973236668 |

The stable object is therefore the tangent line, while its orientation alternates from one elementary stroke to the next.

## Mechanical sequence inside one stroke

After phase-aligning the strongest contraction to phase zero, both regimes show the same ordered sequence:

1. strongest contraction;
2. strongest local foot, measured by maximum absolute local-x tangent amplitude;
3. strongest release.

| Q | Local-foot phase | Release phase | Contraction-to-release distance |
|---:|---:|---:|---:|
| 522.25 | 0.285416667 | 0.443750000 | 0.099227430556 rotations |
| 550 | 0.289583333 | 0.452083333 | 0.102974537037 rotations |

The elementary geometry changes very little between the near-curtain and beyond-curtain regimes. The principal change is the net logarithmic growth retained after large positive and negative contributions nearly cancel.

## Per-stroke growth balance

| Q | OPEN | PHASE | KAPPA | FULL residual growth |
|---:|---:|---:|---:|---:|
| 522.25 | -1.144521078849e-02 | +1.035085702139e-02 | +1.175703029121e-03 | +8.134926202330e-05 |
| 550 | -1.039705253315e-02 | +1.011834891882e-02 | +9.386103765504e-04 | +6.599067622173e-04 |

The branch-closure residual is at machine precision in both cases.

## Evidence chain

- `01_HOPF_CLOSURE`: local threshold law, saturated amplitude, and long-transient interpretation.
- `02_LOCAL_ENERGY`: support-to-local work and energy-balance audit at Q=550.
- `03_FEEDBACK_LOOP`: replay opening, analytic tangent branch split, and continuous feedback-gain curtain.
- `04_CURTAIN_VS_Q`: motion of the neutral curtain with Q, including the focused near-threshold scan.
- `05_PHASE_RESOLVED_TANGENT`: growth choreography and tangent-norm carrier across a reference cycle.
- `06_ROTATING_MEMORY`: reparameterization on the true rotating clock, recurrence, handoff, and mother-form diagnostics.
- `07_SINGLE_WHIP`: the isolated elementary tangent-line stroke and its antiperiodic orientation reversal.
- `08_SCRIPTS`: source code that was actually present among the uploaded files.
- `09_PROVENANCE`: preserved command/output capture and source-status notes.

## Interpretation discipline

The following are numerically supported descriptions:

- recurrent tangent-line geometry;
- orientation reversal after one line period;
- orientation recovery after two line periods;
- nearly unchanged contraction-foot-release sequence at Q=522.25 and Q=550;
- instability requiring closure of the local-to-support feedback loop;
- PHASE as the sign-changing return, with KAPPA reducing the required phase-return gain.

The following remain interpretations or open hypotheses:

- the phrase “mechanical whip” as a physical metaphor for the tangent choreography;
- a projective or Möbius-like global topology;
- universality of the line period over a broad Q range;
- direct identification of the tangent-line stroke with a global winding structure.

See `CLAIMS_LEDGER.csv` for the complete separation between evidence, interpretation, and open questions.

## Integrity

Use `manifest_sha256.txt` at the archive root to verify every preserved file.
