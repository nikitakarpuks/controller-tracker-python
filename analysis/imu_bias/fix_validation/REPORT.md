# Validation A/B of the loader + lever-arm fix (investigator V, 2026-09-23)

(The investigator could not write this file itself; this is its returned report, condensed. Scripts and
CSVs are in this folder and in results/.)

## Bottom line
- Lever arm: use the factory vector read in the correct frame, -R_acc^T t_acc (JSON R as loaded). Today's vector
  (main.py, raw factory t) is WORSE than no lever arm at all.
- Loader + lever together: position error at a 1 s gap 399 -> 137 mm (left, -66 %) and 548 -> 234 mm (right,
  -57 %); on real tracking-loss gaps 221 -> 110 mm (-50 %) and 192 -> 72 mm (-63 %).
- Keep the isotropic accel scale 0.980665 (derived from the driver constant, not fitted).
- Recorder provenance strongly supported, not proven.

## Sanity gates (independent harness)
| Gate | Expected | Reproduced |
|---|---|---|
| IMU-point accel error at 1 s, old loader | ~196 mm | 198.9 mm |
| Same, fully fixed | 75.1 mm (audit) | 75.1 mm |
| Gyro error at 1 s, old loader | 1.70 deg | 1.712 deg |
| LED-origin accel at 1 s, static_dark, old loader + factory lever | 0.42 / 0.44 m | 0.393 / 0.441 m |

## 1. Lever A/B (median error mm, new loader, six recordings pooled; fixed candidate vectors, no fitting leakage)
| ctrl | candidate | 0.1 s | 0.3 s | 1.0 s | real gaps |
|---|---|---|---|---|---|
| left | zero | 15.7 | 63.8 | 231 | 172 |
| left | factory t (what main.py used) | 18.5 | 86.9 | 331 | 216 |
| left | bridge.t | 11.2 | 32.2 | 137.2 | 111 |
| left | **-R^T t (shipped)** | 11.2 | 32.0 | 137.5 | 110 |
| left | old wrong -R_b^T t_b | 11.2 | 32.2 | 134.5 | 106 |
| left | naive constant velocity | 23.9 | 121 | 525 | 438 |
| right | zero | 23.3 | 99.7 | 387 | 141 |
| right | factory t (what main.py used) | 27.4 | 132 | 476 | 179 |
| right | bridge.t | 16.6 | 50.1 | 234.2 | 70.8 |
| right | **-R^T t (shipped)** | 16.3 | 48.9 | 233.7 | 72.0 |
| right | old wrong -R_b^T t_b | 16.6 | 50.2 | 233.4 | 78.9 |
| right | naive constant velocity | 35.0 | 185 | 934 | 282 |

- bridge.t vs -R^T t at 1 s: left +0.37 mm [0.01, 0.71], right +2.59 mm [1.64, 3.48] (paired, block bootstrap) --
  at most 2 % of the error; the 2.6/3.9 mm difference is immaterial. Today's t vs -R^T t: -R^T t better by 223 mm
  (left), 307 mm (right) at 1 s.
- The old wrong -R_b^T t_b is worse on real gaps by 4.9 mm (left) and 7.2 mm (right), CIs exclude 0 (the data only
  weakly discriminate the x sign).
- No other reading of the factory vector works (median mm at 1 s, left/right): -R^T t 135/228 (only one that helps),
  zero 228/378, today's t 337/464, R^T t 395/595, -R t 305/449, R t 269/384, D(-R^T t) 388/587, D t 260/393.

## 2. Attribution (median mm at 1.0 s | real gaps, lever = -R^T t)
| ctrl | variant | 1.0 s | real gaps |
|---|---|---|---|
| left | OLD | 398.7 | 221.4 |
| left | loader-only | 331.2 | 216.1 |
| left | lever-only | 233.8 | 115.9 |
| left | both | 137.5 | 110.0 |
| left | drop 2nd correction only + lever | 184.9 | 107.7 |
| left | scale only + lever | 171.2 | 112.5 |
| right | OLD | 548.2 | 192.3 |
| right | loader-only | 475.6 | 178.9 |
| right | lever-only | 303.9 | 78.3 |
| right | both | 233.7 | 72.0 |
| right | drop 2nd correction only + lever | 257.7 | 74.6 |
| right | scale only + lever | 263.0 | 79.5 |

On the accelerometer point itself (mocap truth, no lever involved), loader alone at 1 s: OLD 198.9, drop 2nd
correction only 135.8, scale only 128.8, both 75.1 mm (-62 %, paired CI [-115, -107] mm). Gyro rotation error
(only the double correction matters): 0.1 s 0.836 -> 0.801, 0.3 s 1.174 -> 1.034, 1.0 s 1.712 -> 1.361 deg
(paired CI [-0.40, -0.31]).

## 3. Gravity / scale reconciliation
Quiet mean |f| (m/s^2, min..max) by stream: old (double-corrected) 10.237..10.284 (+4.4..+4.9 %); scale-only on the old
stream 10.040..10.085 (+2.4..+2.8 %, the earlier "+2.5-3 %"); both fixes 9.810..9.929 (+0.03..+1.2 %). The remainder
after both fixes is the real per-session bias (the audit's model with bias predicts quiet |C| within 0.02 m/s^2 on the
left; with b = 0 it under-predicts by ~0.06). Isotropic 0.980665 is right per the audit's long-window fits (per-axis
residuals left (-0.0002,-0.0013,+0.0011), right (-0.0020,+0.0041,+0.0052), SE 0.0015-0.0018; the right controller has
a 2-3.5 sigma anisotropy one constant cannot remove). Pinned band (median |accel| over samples with |gyro| < 0.5 rad/s,
all 16 pairs): new loader 9.752..9.963 -> [9.70, 10.02] (implemented as ACCEL_QUIET_MAGNITUDE_BAND); old loader
10.13..10.29.

## 4. Recorder provenance
Only `monado-main` (not a git repo, file dates Aug 30) records controller IMUs (wmr_config.h:22-23 indexes 1 and 2;
wmr_controller_base.c:511 pushes the sample to the euroc recorder), taken from wmr_controller_hp_packet_parse in the order
mix, bias, P_oxr rotation (wmr_controller_hp.c:258-276). monado-dev-constellation-controller-tracking records only
imu0. Data evidence: the Wahba angle between recorded gyro axes and the plain flip D is 0.41-1.58 deg on all 16 pairs
across 8 recordings; a raw stream would sit 136.6-138.5 deg away (factory Rt ~105 deg). Not proven: recordings are 4 days
older than the monado-main files and no git metadata exists.

## Caveats
One ~25 min session, one controller pair, six moderate recordings (hard ones used only for provenance). Vision truth
(3-5 mm noise) uses dev-only mocap headset ego-motion; v0 comes from a ~60 ms vision fit; paired comparisons are
unaffected. Gap samples are autocorrelated (5 s block bootstrap); 189-221 real gaps per controller. Remaining right-
controller anisotropy, per-session biases and gyro gain K are untouched by this fix.
