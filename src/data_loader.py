"""
Data Loader
===========

Unified loader for gait data from (in priority order):
1. Consolidated CSV files (data/processed/{patient}/RDV1/{trial}.csv)
2. outputData.mat files (pre-compiled MATLAB format)
3. Raw .txt sensor files (7 sensor files per trial)

All sources produce identical data for the processing pipeline.
"""

import numpy as np
import pandas as pd
import scipy.io as sio
from pathlib import Path


TRIAL_NAMES = [
    'Bare_calibration',
    'Bare_fast',
    'Bare_pref',
    'Shoe_calibration',
    'Shoe_fast',
    'Shoe_pref'
]

SENSOR_REORDER = [7, 5, 6, 4, 3, 2, 1]  # Maps device order to body position order


def load_sensor_from_txt(txt_file):
    """Load one sensor from a .txt file"""
    df = pd.read_csv(txt_file, sep='\t', comment='/')
    df.columns = [c.strip() for c in df.columns]
    df = df.iloc[1:]  # Skip first row (matches outputData.mat behavior)

    return {
        'quat':  df[['Quat_q0', 'Quat_q1', 'Quat_q2', 'Quat_q3']].values.astype(np.float64),
        'acc':   df[['Acc_X', 'Acc_Y', 'Acc_Z']].values.astype(np.float64),
        'gyr':   df[['Gyr_X', 'Gyr_Y', 'Gyr_Z']].values.astype(np.float64),
        'mag':   df[['Mag_X', 'Mag_Y', 'Mag_Z']].values.astype(np.float64),
        'euler': df[['Roll', 'Pitch', 'Yaw']].values.astype(np.float64)
    }


def load_trial_from_txt(trial_dir, trial_name):
    """
    Load one trial's data from 7 sensor .txt files

    Parameters:
    -----------
    trial_dir : Path
        Directory containing the 7 .txt files
    trial_name : str
        Name of the trial (e.g. 'Bare_fast')

    Returns:
    --------
    dict with 'Name' and 'cal' (list of 7 sensor dicts)
    """
    trial_dir = Path(trial_dir)
    txt_files = sorted(trial_dir.glob('*.txt'))

    if len(txt_files) != 7:
        raise ValueError(
            f"Expected 7 sensor files in {trial_dir}, found {len(txt_files)}"
        )

    cal_data = [load_sensor_from_txt(f) for f in txt_files]
    return {'Name': trial_name, 'cal': cal_data}


def load_trial_from_mat(mat_trial):
    """Convert a trial from outputData.mat format to standard dict"""
    cal_data = []
    for sensor in mat_trial['cal']:
        cal_data.append({
            'quat':  np.array(sensor['quat'], dtype=np.float64),
            'acc':   np.array(sensor['acc'],  dtype=np.float64),
            'gyr':   np.array(sensor['gyr'],  dtype=np.float64),
            'mag':   np.array(sensor['mag'],  dtype=np.float64),
            'euler': np.array(sensor['euler'], dtype=np.float64)
        })
    return {'Name': mat_trial['Name'], 'cal': cal_data}


def apply_sensor_reorder(trial, new_order=SENSOR_REORDER):
    """
    Reorder sensors from device order to body position order.

    Body positions after reorder:
    [0] waist_IMU
    [1] Non-affected Thigh
    [2] Non-affected Shank
    [3] Non-affected Foot
    [4] Affected Thigh
    [5] Affected Shank
    [6] Affected Foot
    """
    reordered_cal = [trial['cal'][i - 1] for i in new_order]
    return {'Name': trial['Name'], 'cal': reordered_cal}


def load_trial_from_csv(csv_path, trial_name):
    """
    Load one trial from a consolidated CSV file.

    The CSV has columns: sensor, sample, quat_w/x/y/z, acc_x/y/z, etc.
    with all 7 sensors stacked.
    """
    df = pd.read_csv(csv_path)
    cal_data = []

    for sensor_idx in range(7):
        sdf = df[df['sensor'] == sensor_idx].sort_values('sample')
        cal_data.append({
            'quat':  sdf[['quat_w', 'quat_x', 'quat_y', 'quat_z']].values.astype(np.float64),
            'acc':   sdf[['acc_x', 'acc_y', 'acc_z']].values.astype(np.float64),
            'gyr':   sdf[['gyr_x', 'gyr_y', 'gyr_z']].values.astype(np.float64),
            'mag':   sdf[['mag_x', 'mag_y', 'mag_z']].values.astype(np.float64),
            'euler': sdf[['roll', 'pitch', 'yaw']].values.astype(np.float64)
        })

    return {'Name': trial_name, 'cal': cal_data}


def load_patient_data(patient_id, rdv='RDV1', base_path=None):
    """
    Load all trials for a patient from any available source.

    Priority: CSV (fastest) > outputData.mat > raw .txt files

    Parameters:
    -----------
    patient_id : str
        Patient identifier (e.g. '01-P-AR')
    rdv : str
        Visit identifier (e.g. 'RDV1')
    base_path : Path
        Base path containing Kinematics folder

    Returns:
    --------
    re_output : list of 6 dicts (one per trial, sensors in body-position order)
    source : str ('csv', 'mat', or 'txt')

    Raises:
    -------
    FileNotFoundError if no data source is available
    """
    if base_path is None:
        base_path = Path(__file__).parent.parent / 'Kinematics'

    project_root = base_path.parent
    csv_base = project_root / 'data' / 'processed' / patient_id / rdv
    mat_path = base_path / 'Dataset' / 'hemiparetic' / patient_id / rdv / f'{patient_id}_{rdv}_outputData.mat'
    txt_base = base_path / 'Dataset' / 'hemiparetique' / patient_id / rdv / 'Data'

    # Try consolidated CSV first (fastest)
    if csv_base.exists() and all((csv_base / f'{t}.csv').exists() for t in TRIAL_NAMES):
        trials = [load_trial_from_csv(csv_base / f'{t}.csv', t) for t in TRIAL_NAMES]
        source = 'csv'

    # Try outputData.mat
    elif mat_path.exists():
        mat_data = sio.loadmat(str(mat_path), simplify_cells=True)
        raw_output = mat_data['output']
        trials = [load_trial_from_mat(raw_output[i]) for i in range(6)]
        source = 'mat'

    # Fall back to raw txt files
    elif txt_base.exists():
        missing = [t for t in TRIAL_NAMES if not (txt_base / t).exists() or
                   len(list((txt_base / t).glob('*.txt'))) != 7]
        if missing:
            raise FileNotFoundError(
                f"Incomplete txt data for {patient_id}: missing or incomplete trials {missing}"
            )

        trials = [load_trial_from_txt(txt_base / t, t) for t in TRIAL_NAMES]
        source = 'txt'

    else:
        raise FileNotFoundError(
            f"No data found for {patient_id} {rdv}: "
            f"no CSV, outputData.mat, or txt files"
        )

    # Apply sensor reorder (device order -> body position order)
    re_output = [apply_sensor_reorder(t) for t in trials]

    return re_output, source


def load_index_files(patient_id, rdv='RDV1', base_path=None):
    """
    Load all 6 index files for a patient.

    Priority: CSV (data/processed/) > .mat files (Kinematics/)

    Returns:
    --------
    indices : dict mapping trial_name -> index_data (or None if missing)
        Each index_data has 'indexStart' and 'indexEnd' keys.
    """
    if base_path is None:
        base_path = Path(__file__).parent.parent / 'Kinematics'

    project_root = base_path.parent
    csv_path = project_root / 'data' / 'processed' / patient_id / rdv / 'indices.csv'

    indices = {}

    # Try CSV first
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        for trial in TRIAL_NAMES:
            tdf = df[df['trial'] == trial]
            if len(tdf) == 0:
                indices[trial] = None
                continue
            # For walking trials (section 0 only), use first section
            # For calibration trials with multiple sections, use first section
            row = tdf[tdf['section'] == 0].iloc[0]
            indices[trial] = {
                'indexStart': int(row['index_start']),
                'indexEnd': int(row['index_end'])
            }
        return indices

    # Fall back to .mat files
    for search_dir in [
        base_path / 'Dataset' / 'hemiparetic' / patient_id / rdv,
        base_path / 'Dataset' / 'hemiparetic' / 'Ne pas utiliser' / patient_id / rdv
    ]:
        if not search_dir.exists():
            continue
        for trial in TRIAL_NAMES:
            if trial in indices:
                continue
            idx_file = search_dir / f'{trial}_indexData.mat'
            if idx_file.exists():
                mat = sio.loadmat(str(idx_file), simplify_cells=True)
                indices[trial] = mat

    # Fill missing trials with None
    for trial in TRIAL_NAMES:
        if trial not in indices:
            indices[trial] = None

    return indices


def check_patient_availability(patient_id, rdv='RDV1', base_path=None):
    """
    Check what data is available for a patient without loading it.

    Returns:
    --------
    dict with keys: has_mat, has_txt, has_full_txt, has_indices, processable
    """
    if base_path is None:
        base_path = Path(__file__).parent.parent / 'Kinematics'

    project_root = base_path.parent
    csv_base = project_root / 'data' / 'processed' / patient_id / rdv
    mat_path = base_path / 'Dataset' / 'hemiparetic' / patient_id / rdv / f'{patient_id}_{rdv}_outputData.mat'
    txt_base = base_path / 'Dataset' / 'hemiparetique' / patient_id / rdv / 'Data'
    idx_dir = base_path / 'Dataset' / 'hemiparetic' / patient_id / rdv

    has_csv = csv_base.exists() and all((csv_base / f'{t}.csv').exists() for t in TRIAL_NAMES)
    has_mat = mat_path.exists()
    has_txt = txt_base.exists()
    has_full_txt = has_txt and all(
        len(list((txt_base / t).glob('*.txt'))) == 7
        for t in TRIAL_NAMES
    )
    has_indices = idx_dir.exists()

    if has_csv:
        source = 'csv'
    elif has_mat:
        source = 'mat'
    elif has_full_txt:
        source = 'txt'
    else:
        source = 'none'

    return {
        'has_csv': has_csv,
        'has_mat': has_mat,
        'has_txt': has_txt,
        'has_full_txt': has_full_txt,
        'has_indices': has_indices,
        'processable': has_indices and (has_csv or has_mat or has_full_txt),
        'source': source
    }
