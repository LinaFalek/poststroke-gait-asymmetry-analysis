# Poststroke Gait Asymmetry Analysis

Python pipeline for extracting and analyzing gait parameters from IMU sensor data in hemiparetic patients. The pipeline replicates and validates a reference implementation, with subject-specific parameter optimization for swing phase detection.

## Dataset

- **26 hemiparetic patients** (poststroke), up to 5 follow-up visits (RDV1–RDV5)
- **4 walking conditions** per visit: barefoot preferred speed, barefoot fast, shod preferred speed, shod fast
- **7 IMU sensors**: waist, thighs / shanks / feet


## Method

1. **Swing phase detection** — foot acceleration norm thresholded by 5 parameters `[Dh, Dl, Ds, Tm, Td]`
2. **Subject-specific tuning** — optimized via Nelder-Mead with physiological bounds, minimizing mean relative error vs. MATLAB ground truth
3. **Gait parameter extraction** — spatiotemporal (stride length/time, stance/swing duration) and joint angles (hip, knee, ankle — max/min in stance and swing) for affected and non-paretic sides


