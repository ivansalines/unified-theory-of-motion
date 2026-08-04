# Reproducibility Status

## Fully preserved from the uploaded materials

- reports;
- summary tables;
- phase and time-series tables;
- compressed raw arrays that were uploaded;
- console logs;
- figures;
- the phase-resolved tangent-cycle script;
- recorded command lines and parameter values.

## Source files not present among the uploaded materials

The following original inputs or scripts were referenced by reports and logs but were not available in the active uploaded-file set:

- `q550_bistability_states_1000T.npz`;
- `support_local_energy_audit_q550_raw.npz`;
- `support_feedback_q_curtain_focus_cache/q_522p25_orbit.npz`;
- `support_feedback_q_curtain_cache/q_550_orbit.npz`;
- the original source scripts for several upstream audits;
- the source scripts for the rotating-memory and single-whip audits.

No missing source file has been fabricated or silently reconstructed. The archive preserves the available derived arrays, reports, logs, and exact recorded parameters so that the milestone remains auditable.

## Recorded commands for the final two analysis stages

Rotating-memory audit:

```bash
python3.13 test_support_rotating_phase_memory_audit.py \
  --analysis-rotations 60 \
  --steps-per-period 160 \
  --phase-bins 180 \
  2>&1 | tee support_rotating_memory_audit_console.log
```

Single-whip audit:

```bash
python3.13 test_support_single_whip_tangent_audit.py \
  --tail-periods 70 \
  --steps-per-period 160 \
  --samples-per-rotation 720 \
  --segment-count 120 \
  --stroke-bins 240 \
  2>&1 | tee support_single_whip_audit_console.log
```

Phase-resolved tangent-cycle audit:

```bash
python3.13 test_support_tangent_cycle_audit.py \
  --analysis-periods 80 \
  --steps-per-period 160 \
  2>&1 | tee support_tangent_cycle_audit_console.log
```
