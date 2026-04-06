# Poststroke Gait Asymmetry Analysis

Python pipeline for extracting and analyzing gait parameters from IMU sensor data in hemiparetic patients. The goal is to characterize walking asymmetry between the affected and non-paretic sides across multiple sessions and conditions.

## Dataset

- **26 hemiparetic patients** (poststroke), up to 5 follow-up visits (RDV1–RDV5)
- **27 healthy control subjects**, single session
- **4 walking conditions** per session: barefoot preferred speed, barefoot fast, shod preferred speed, shod fast
- **7 IMU sensors** placed on the waist, thighs, shanks, and feet

The dataset is not included in the repository. Place it under `Kinematics/Dataset/`, keeping the original folder structure. The pipeline expects one subfolder per patient, with subfolders per visit for the hemiparetic group. Patient IDs follow the format `{XX}-P-{YY}` for hemiparetic and `{XX}-T-{YY}` for control subjects.

## Workflow

### Step 1 — Swing detection parameter tuning

The swing phase detection relies on 5 parameters `[Dh, Dl, Ds, Tm, Td]` that govern a finite-state machine applied to foot acceleration signals. These are tuned per subject using Nelder-Mead optimization against reference gait parameters, using the first visit only. The resulting parameters are then applied to all subsequent visits.

```
python scripts/tune_swing_parameters.py          # hemiparetic group
python scripts/tune_swing_parameters_controle.py # control group
```

Output:
```
output/hemiparetic_tuned/
    fine_tuning_summary.csv
output/controle_tuned/
    fine_tuning_summary.csv
```

### Step 2 — Joint-angle CSV export

Runs the full gait pipeline using the tuned parameters and exports kinematics for each trial.

```
python scripts/export_joint_angles_hemiparetic.py  # all patients × RDV1–5 × 4 trials
python scripts/export_joint_angles_controle.py     # all controls × 4 trials
```

Each trial folder contains five files:

| File | Content |
|------|---------|
| `summary.csv` | Number of detected strides per side |
| `events.csv` | Stride-level gait events and timing |
| `angles_time_series.csv` | Continuous joint-angle signals over the full trial (hip, knee, ankle — both sides) |
| `angles_per_stride_raw.csv` | Joint angles segmented stride by stride, original sample resolution |
| `angles_per_stride_normalized.csv` | Stride-level angles normalized to 0–100% gait cycle |

Output:
```
output/csv_exports/
    hemiparetic/
        {patient_id}/
            {RDV}/
                {trial}/
                    summary.csv
                    events.csv
                    angles_time_series.csv
                    angles_per_stride_raw.csv
                    angles_per_stride_normalized.csv
    controle/
        {patient_id}/
            {trial}/
                summary.csv  ...
output/csv_exports_merged/
    hemiparetic/
        summary.csv
        events.csv
        angles_time_series.csv
        angles_per_stride_raw.csv
        angles_per_stride_normalized.csv
    controle/
        summary.csv  ...
```

All rows include `patient_id`, `rdv`, and `trial_name` columns for identification.

### Step 3 — Gait visualization

Gait cycles and spatiotemporal summaries can be plotted for any patient and visit:

```
python scripts/generate_gait_plots.py
```

Figures are saved under `output/figures/`.
