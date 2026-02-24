# Poststroke Gait Asymmetry Analysis

Python pipeline for extracting and analyzing gait parameters from IMU sensor data in hemiparetic patients. The pipeline replicates and validates a reference implementation, with subject-specific parameter optimization for swing phase detection.

## Dataset

- **26 hemiparetic patients** (poststroke), up to 5 follow-up visits (RDV1–RDV5)
- **4 walking conditions** per visit: barefoot preferred speed, barefoot fast, shod preferred speed, shod fast
- **7 IMU sensors**: waist, bilateral thigh / shank / foot
- Sensor reorder: `[7, 5, 6, 4, 3, 2, 1]` → `[waist | NonAff thigh, shank, foot | Aff thigh, shank, foot]`


## Method

1. **Swing phase detection** — foot acceleration norm thresholded by 5 parameters `[Dh, Dl, Ds, Tm, Td]`
2. **Subject-specific tuning** — optimized via Nelder-Mead with physiological bounds, minimizing mean relative error vs. MATLAB ground truth
3. **Gait parameter extraction** — spatiotemporal (stride length/time, stance/swing duration) and joint angles (hip, knee, ankle — max/min in stance and swing) for affected and non-paretic sides

## Repository Structure

```
├── src/
│   ├── gait_processing.py      # Core pipeline: swing detection + parameter extraction
│   ├── data_loader.py          # Loads raw sensor txt files or pre-processed CSVs
│   └── gait_functions.py       # Signal processing utilities
│
├── scripts/
│   ├── tune_swing_parameters.py  # Optimize detection parameters per patient/condition
│   └── generate_gait_plots.py    # Batch plot generation (joint angles, foot accel, spatiotemporal)
│
└── results/
    ├── hemiparetic_tuned/      # Per-patient GaitSummary CSVs, Python vs MATLAB comparison,
    │                           #   condition_params.csv (tuned bare/shoe params + errors)
    └── gait_plots/             # Gait cycle figures — all patients × visits × conditions
                                #   01_joint_angles.png | 02_foot_acceleration.png | 03_spatiotemporal.png
```

