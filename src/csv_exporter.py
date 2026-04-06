
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d


N_NORM = 200  # number of points for normalized gait cycle (0-100%)


def export_trial_csvs(result, patient_id, rdv, trial_name, out_dir):
    """
    Export 5 CSV files from a process_gait_data result.
    result must be obtained with return_debug=True.

    Files written:
        summary.csv
        events.csv
        angles_time_series.csv
        angles_per_stride_raw.csv
        angles_per_stride_normalized.csv
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dbg            = result['_debug']
    angles         = dbg['angles']
    swing          = dbg['swing']
    step_time_data = dbg['step_time_data']
    time_vec       = dbg['time_vec'] / 1000.0   # ms -> s

    aa, ak, ah = angles['aa'], angles['ak'], angles['ah']
    na, nk, nh = angles['na'], angles['nk'], angles['nh']
    sw_af = swing['af']
    sw_nf = swing['nf']

    step_af   = step_time_data['af']
    step_nf   = step_time_data['nf']
    n_samples = len(time_vec)

    side_configs = [
        ('affected',    step_af, aa, ak, ah),
        ('non_paretic', step_nf, na, nk, nh),
    ]

    # ── 1. summary.csv ──────────────────────────────────────────────────────
    summary_rows = [
        {'patient_id': patient_id, 'rdv': rdv, 'trial_name': trial_name,
         'side': 'affected',    'n_strides': len(step_af) if len(step_af) > 0 else 0},
        {'patient_id': patient_id, 'rdv': rdv, 'trial_name': trial_name,
         'side': 'non_paretic', 'n_strides': len(step_nf) if len(step_nf) > 0 else 0},
    ]
    pd.DataFrame(summary_rows).to_csv(out_dir / 'summary.csv', index=False)

    # ── 2. events.csv ────────────────────────────────────────────────────────
    event_rows = []
    for side_label, step_data, *_ in side_configs:
        if len(step_data) == 0:
            continue
        for i, row in enumerate(step_data):
            s0, s1, s2 = int(row[0]), int(row[1]), int(row[2])
            if s2 >= n_samples:
                continue
            event_rows.append({
                'patient_id':               patient_id,
                'rdv':                      rdv,
                'trial_name':               trial_name,
                'side':                     side_label,
                'stride_index':             i,
                'stance_start_idx':         s0,
                'swing_start_idx':          s1,
                'next_stance_start_idx':    s2,
                'stance_start_time_s':      time_vec[s0],
                'swing_start_time_s':       time_vec[s1],
                'next_stance_start_time_s': time_vec[s2],
                'stance_samples':           s1 - s0,
                'swing_samples':            s2 - s1,
                'stride_samples':           s2 - s0,
            })
    pd.DataFrame(event_rows).to_csv(out_dir / 'events.csv', index=False)

    # ── 3. angles_time_series.csv ────────────────────────────────────────────
    pd.DataFrame({
        'patient_id':                  patient_id,
        'rdv':                         rdv,
        'trial_name':                  trial_name,
        'sample_index':                np.arange(n_samples),
        'time_s':                      time_vec,
        'swing_affected':              sw_af.astype(int),
        'swing_non_paretic':           sw_nf.astype(int),
        'angle_affected_ankle_deg':    aa,
        'angle_affected_knee_deg':     ak,
        'angle_affected_hip_deg':      ah,
        'angle_non_paretic_ankle_deg': na,
        'angle_non_paretic_knee_deg':  nk,
        'angle_non_paretic_hip_deg':   nh,
    }).to_csv(out_dir / 'angles_time_series.csv', index=False)

    # ── 4. angles_per_stride_raw.csv ─────────────────────────────────────────
    raw_rows = []
    for side_label, step_data, ankle, knee, hip in side_configs:
        if len(step_data) == 0:
            continue
        for i, row in enumerate(step_data):
            s0, s1, s2 = int(row[0]), int(row[1]), int(row[2])
            if s2 >= n_samples:
                s2 = n_samples - 1
            stride_len = s2 - s0
            stance_len = s1 - s0
            swing_len  = s2 - s1
            if stride_len <= 0:
                continue
            for j in range(s0, s2):
                if j >= n_samples:
                    break
                gc_pct = (j - s0) / stride_len * 100
                if j < s1:
                    phase     = 'stance'
                    phase_pct = (j - s0) / stance_len * 100 if stance_len > 0 else 0.0
                else:
                    phase     = 'swing'
                    phase_pct = (j - s1) / swing_len  * 100 if swing_len  > 0 else 0.0
                raw_rows.append({
                    'patient_id':     patient_id,
                    'rdv':            rdv,
                    'trial_name':     trial_name,
                    'side':           side_label,
                    'stride_index':   i,
                    'sample_index':   j,
                    'time_s':         time_vec[j],
                    'phase':          phase,
                    'phase_pct':      phase_pct,
                    'gait_cycle_pct': gc_pct,
                    'ankle_deg':      ankle[j],
                    'knee_deg':       knee[j],
                    'hip_deg':        hip[j],
                })
    pd.DataFrame(raw_rows).to_csv(out_dir / 'angles_per_stride_raw.csv', index=False)

    # ── 5. angles_per_stride_normalized.csv ──────────────────────────────────
    gc_norm   = np.linspace(0, 100, N_NORM)
    norm_rows = []
    for side_label, step_data, ankle, knee, hip in side_configs:
        if len(step_data) == 0:
            continue
        for i, row in enumerate(step_data):
            s0, s2 = int(row[0]), int(row[2])
            if s2 >= n_samples:
                s2 = n_samples - 1
            length = s2 - s0
            if length < 2:
                continue
            gc_orig  = np.linspace(0, 100, length)
            a_interp = interp1d(gc_orig, ankle[s0:s2], kind='linear')(gc_norm)
            k_interp = interp1d(gc_orig, knee[s0:s2],  kind='linear')(gc_norm)
            h_interp = interp1d(gc_orig, hip[s0:s2],   kind='linear')(gc_norm)
            for j in range(N_NORM):
                norm_rows.append({
                    'patient_id':     patient_id,
                    'rdv':            rdv,
                    'trial_name':     trial_name,
                    'side':           side_label,
                    'stride_index':   i,
                    'gait_cycle_pct': gc_norm[j],
                    'ankle_deg':      a_interp[j],
                    'knee_deg':       k_interp[j],
                    'hip_deg':        h_interp[j],
                })
    pd.DataFrame(norm_rows).to_csv(out_dir / 'angles_per_stride_normalized.csv', index=False)
