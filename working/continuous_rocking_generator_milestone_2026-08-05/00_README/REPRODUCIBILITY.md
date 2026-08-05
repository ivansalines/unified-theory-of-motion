# Reproducibility

## Reproduce the continuous-generator analysis

From the archive root:

```bash
cd 04_CONTINUOUS_GENERATOR

python3.13 test_support_stride_generator_audit.py \
  --input ../03_STRIDE_INVOLUTION/support_stride_involution_audit_raw.npz \
  2>&1 | tee reproduced_support_stride_generator_audit_console.log
```

This step requires Python 3.13 with NumPy, SciPy, and Matplotlib available.
It does not regenerate nonlinear orbits.

## Reproduce the stride-involution analysis

The included stride-involution script depends on the validated monodromy
audit implementation and on two nonlinear orbit caches:

- `support_feedback_q_curtain_focus_cache/q_522p25_orbit.npz`
- `support_feedback_q_curtain_cache/q_550_orbit.npz`

Those orbit caches are not present in the active archive source set and are
therefore not included. The exported stride raw archive, reports, tables,
figures, and console log are preserved.

## Reproduce the direct B(s) embedding

The included direct-embedding script also requires the two orbit caches above
and the monodromy audit script.

## Reproducibility status

- Continuous-generator audit from preserved stride raw NPZ: self-contained
  apart from the Python scientific stack.
- Stride-involution audit from nonlinear orbit caches: source code preserved,
  orbit caches missing.
- Direct B(s) embedding from nonlinear orbit caches: source code preserved,
  orbit caches missing.
- Original monodromy audit from nonlinear orbit caches: source code preserved,
  orbit caches missing.

No missing file has been reconstructed or represented as an original.
