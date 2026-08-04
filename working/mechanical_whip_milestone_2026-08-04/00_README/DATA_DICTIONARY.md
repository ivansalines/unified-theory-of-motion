# Data Dictionary

## Main scalar quantities

- `Q`: control parameter used by the model.
- `T_ref`: reference period, approximately 12.566370614359.
- `Theta`: rotating-support phase coordinate.
- `s = Theta/(2*pi)`: unwrapped support-rotation coordinate.
- `eta_phase`: diagnostic gain multiplying the local-to-phase return path.
- `eta_kappa`: diagnostic gain multiplying the local-to-kappa return path.
- `lambda`: logarithmic tangent growth per reference period unless another clock is explicitly stated.
- `H_total`: instantaneous logarithmic growth density per complete support rotation.
- `H_open`, `H_phase`, `H_kappa`: exact tangent-growth contributions whose sum is `H_total`.

## Tangent observables

- `delta_x`: local tangent displacement.
- `delta_Theta`: rotating-support phase tangent displacement.
- `delta_Delta`: tangent displacement of the relative phase combination.
- `delta_rho1`, `delta_rho2`: physical radial tangent displacements.
- `delta_rho_sym`: symmetric radial tangent combination.
- `delta_rho_asym`: antisymmetric radial tangent combination.
- `delta_activity`: tangent displacement of the support activity term.
- `delta_kappa`: tangent displacement of the effective kappa coupling.

## Recurrence quantities

- `line_overlap_absolute`: mean absolute tangent overlap. It treats `w` and `-w` as the same tangent line.
- `signed_overlap`: mean oriented tangent overlap.
- `line_period_rotations`: first nonzero recurrence of the tangent line on the support-rotation clock.
- `oriented_period_rotations`: first recurrence that restores both the line and its orientation.
- `stroke_phase`: normalized phase inside one tangent-line recurrence interval.

## Folder-specific files

### `05_PHASE_RESOLVED_TANGENT`

- `*_phase_profiles.csv`: mean and standard deviation of phase-conditioned growth contributions, tangent-norm shares, and physical tangent observables.
- `*_periods.csv`: period-by-period growth estimates.
- `*_raw.npz`: compressed numerical arrays used for the phase-resolved figures.

### `06_ROTATING_MEMORY`

- `*_continuity.csv`: signed and absolute tangent overlaps versus lag in support rotations.
- `*_phase_profiles.csv`: rotating-phase profiles of growth, observables, tangent-norm shares, and consecutive-rotation overlap.
- `*_rotations.csv`: growth carried by individual complete support rotations.

### `07_SINGLE_WHIP`

- `*_recurrence.csv`: recurrence curves expressed in both support rotations and tangent-line strokes.
- `*_profiles.csv`: line-aligned signed components and invariant growth contributions within one elementary stroke.
- `*_segments.csv`: stroke-by-stroke orientation flips and overlap with the preceding stroke.
- `*_summary.csv`: compact table of line periods, overlaps, growth balances, and component peak phases.
