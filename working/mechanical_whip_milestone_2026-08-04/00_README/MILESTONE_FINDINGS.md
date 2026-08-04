# Milestone Findings

## 1. Closed feedback is essential

At Q=550, the fully closed system has a positive tangent exponent, while replaying the recorded support waveform into an opened local subsystem produces strong conditional damping. The instability is therefore not caused by a prescribed external waveform alone; it requires closure of the local-support feedback loop.

Primary evidence:

- `03_FEEDBACK_LOOP/support_feedback_full_replay_mean_q550_report.txt`
- `03_FEEDBACK_LOOP/support_feedback_full_replay_mean_q550_summary.csv`

## 2. The phase-return branch changes the sign

The validated analytic tangent split at Q=550 gives:

- OPEN: approximately -1.885 per reference period;
- PHASE_ONLY: approximately +2.995e-03;
- KAPPA_ONLY: approximately -4.411e-02;
- FULL: approximately +6.182e-03.

The phase-return path is sufficient to create a positive mode. The kappa-return path is not sufficient by itself, but it modifies the phase-driven instability and lowers the critical phase-return gain.

Primary evidence:

- `03_FEEDBACK_LOOP/support_feedback_branch_split_q550_tangent_report.txt`
- `03_FEEDBACK_LOOP/support_feedback_gain_map_q550_report.txt`

## 3. The neutral curtain moves continuously

At Q=550, a neutral curve exists across the full sampled kappa-return gain range. Increasing the kappa-return gain lowers the critical phase-return gain. The fitted boundary is nearly quadratic over the sampled domain.

The Q-dependent curtain analysis shows that the top edge is pierced near the direct local-growth threshold only when the kappa-return path is nearly fully active. This demonstrates that the observed onset is a feedback-closure event, not merely the crossing of an isolated local coefficient.

Primary evidence:

- `03_FEEDBACK_LOOP/support_feedback_gain_map_q550_report.txt`
- `04_CURTAIN_VS_Q/support_feedback_curtain_vs_q_report.txt`
- `04_CURTAIN_VS_Q/support_feedback_curtain_focus_report.txt`

## 4. The tangent mode is carried mainly by the rotating sector

Across both Q=522.25 and Q=550, approximately 99.6 percent of the weighted tangent norm is carried by the rotating variables. The local and radial components remain small in norm but participate in the timing of the handoff.

Primary evidence:

- `05_PHASE_RESOLVED_TANGENT/support_tangent_cycle_audit_report.txt`
- `06_ROTATING_MEMORY/support_rotating_memory_audit_report.txt`

## 5. One support rotation is not the intrinsic recurrence clock

Comparing tangent directions exactly one support rotation apart produces low overlap because several elementary tangent-line strokes occur inside one support rotation. The intrinsic recurrence is approximately 0.224 to 0.228 support rotations.

Primary evidence:

- `06_ROTATING_MEMORY/support_rotating_memory_audit_continuity.csv`
- `07_SINGLE_WHIP/support_single_whip_audit_recurrence.csv`

## 6. The tangent line is antiperiodic as an oriented vector

For both retained Q values, one elementary recurrence returns the same tangent line with opposite orientation:

\[
w(s+P_{\mathrm{line}}) \approx -w(s).
\]

Two elementary recurrences restore the oriented vector:

\[
w(s+2P_{\mathrm{line}}) \approx w(s).
\]

The line overlap and signed overlap differ only by sign at the first recurrence and are both nearly unity in magnitude. Exactly half of the retained strokes require a sign flip when aligned as an unoriented line.

Primary evidence:

- `07_SINGLE_WHIP/support_single_whip_audit_report.txt`
- `07_SINGLE_WHIP/support_single_whip_audit_summary.csv`
- `07_SINGLE_WHIP/support_single_whip_audit_segments.csv`

## 7. The same elementary stroke appears in both regimes

The line period changes by less than two percent from Q=522.25 to Q=550. The local-foot phase and release phase are also nearly unchanged. The near-curtain and beyond-curtain regimes therefore share the same elementary kinematic grammar.

The principal difference is the residual growth retained after the OPEN, PHASE, and KAPPA contributions cancel. The beyond-curtain case does not require a qualitatively different stroke; it retains a larger positive remainder of the same stroke.

## 8. Cautions

- The antiperiodic recurrence is demonstrated for the analyzed tangent metric and the two cached central orbits.
- A Möbius-like or projective interpretation is suggestive but not formally established.
- Near the threshold, the sign of a finite-window mean growth rate can be sensitive to the retained window because the per-rotation fluctuations are much larger than the residual mean.
- Several source scripts and the original central-orbit cache files were not present among the uploaded files. The archive preserves all available outputs and records this reproducibility gap explicitly.
