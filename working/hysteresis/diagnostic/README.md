```
python3.13 test_hysteresis_diagnostic_english_v2.py 
Script version: 2026-07-31-layout-fix-v2
n=10; Q: 400 -> 1200 -> 400
Radial damping gamma_rho = 0.3
Damped x period       = 4.074925
Initial Delta period  = 12.566371
Reference period      = 12.566371
Per Q: 1 reference periods for settling + 2 reference periods for measurement
[  1/299] Q= 400.000  x_mean=0.5003663  x_std=5.664e-03  rho=(1.7430, 0.1570)
[ 29/299] Q= 550.336  x_mean=0.4999876  x_std=2.943e-04  rho=(1.7247, 0.1182)
[ 58/299] Q= 706.040  x_mean=0.5000050  x_std=2.904e-04  rho=(1.7229, 0.1084)
[ 87/299] Q= 861.745  x_mean=0.5000140  x_std=2.510e-04  rho=(1.7217, 0.1031)
[116/299] Q=1017.450  x_mean=0.4999442  x_std=4.917e-04  rho=(1.7202, 0.1006)
[145/299] Q=1173.154  x_mean=0.4999201  x_std=7.869e-04  rho=(1.7196, 0.0985)
[174/299] Q=1071.141  x_mean=0.5000867  x_std=7.423e-04  rho=(1.7219, 0.1009)
[203/299] Q= 915.436  x_mean=0.5000220  x_std=4.632e-04  rho=(1.7215, 0.1014)
[232/299] Q= 759.732  x_mean=0.5000015  x_std=2.245e-04  rho=(1.7219, 0.1062)
[261/299] Q= 604.027  x_mean=0.4999739  x_std=3.313e-04  rho=(1.7234, 0.1138)
[290/299] Q= 448.322  x_mean=0.4999825  x_std=1.957e-04  rho=(1.7259, 0.1302)
[299/299] Q= 400.000  x_mean=0.5000020  x_std=4.978e-05  rho=(1.7285, 0.1398)
Transition candidate on upward branch: Q ≈ 400.000000
Transition indicator strength          = 4.463478e-02
Maximum upward x_std                  = 5.663890e-03 at Q=400.000000
Maximum matched |Δx_mean|             = 3.643320e-04
Maximum matched |Δx_std|              = 5.614111e-03
Diagnostic summary saved to: /nfs-mount/elab1/hysteresis/diagnostic/diagnostic_summary.png
Phase portraits saved to: /nfs-mount/elab1/hysteresis/diagnostic/diagnostic_phase_spirals.png
Adiabatic transition panels saved to: /nfs-mount/elab1/hysteresis/diagnostic/diagnostic_adiabatic_transition_panels.png
```

12 phase spirals
6 on the upward branch,
6 on the downward branch,
color-coded by local time.
12 adiabatic transition panels
each one shows:
settling window,
measurement window,
local x(t) response.
What it is saying already

In the demo, the strongest contraction signal lands at the low-Q entrance, around:

Q?400

So the main transition, with the current parameters, looks like a rapid collapse right at the beginning of the upward sweep, then a long regime of much thinner loops.

Run
python test_hysteresis_diagnostic_english.py

If you want explicit outputs:

python test_hysteresis_diagnostic_english.py \
  --summary-output diagnostic_summary.png \
  --spirals-output diagnostic_phase_spirals.png \
  --transition-output diagnostic_adiabatic_transition_panels.png

