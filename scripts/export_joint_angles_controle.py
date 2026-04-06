"""
Export joint-angle CSV outputs for the control group.

For each control subject / trial, runs the gait pipeline using tuned
swing-detection parameters and writes:
    output/csv_exports/controle/{patient_id}/{trial_name}/
        summary.csv
        events.csv
        angles_time_series.csv
        angles_per_stride_raw.csv
        angles_per_stride_normalized.csv

Usage:
    python scripts/export_joint_angles_controle.py            # all control subjects
    python scripts/export_joint_angles_controle.py 01-T-CC    # specific subject
    python scripts/export_joint_angles_controle.py --force    # overwrite existing
"""

import sys
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings('ignore')

from src.data_loader import apply_sensor_reorder, TRIAL_NAMES, load_trial_from_csv
from src.gait_processing import process_gait_data
from src.csv_exporter import export_trial_csvs
import ast
import pandas as pd
import scipy.io as sio
import numpy as np

MERGE_ROOT = Path(__file__).parent.parent / 'output' / 'csv_exports_merged' / 'controle'

PROJECT_ROOT  = Path(__file__).parent.parent
CONTROLE_DIR  = PROJECT_ROOT / 'Kinematics' / 'Dataset' / 'controle'
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed' / 'controle'
OUT_ROOT      = PROJECT_ROOT / 'output' / 'csv_exports' / 'controle'
TUNED_SUMMARY = PROJECT_ROOT / 'output' / 'controle_tuned' / 'fine_tuning_summary.csv'

DEFAULT_PARAMS = [100, 30, -300, 30, 10]
BARE_TRIALS    = {'Bare_fast', 'Bare_pref'}
WALKING_TRIALS = ['Bare_fast', 'Bare_pref', 'Shoe_fast', 'Shoe_pref']
TRIAL_TO_CAL   = {
    'Bare_fast': 'Bare_calibration', 'Bare_pref': 'Bare_calibration',
    'Shoe_fast': 'Shoe_calibration', 'Shoe_pref': 'Shoe_calibration',
}


def load_tuned_params():
    if not TUNED_SUMMARY.exists():
        return {}
    df = pd.read_csv(TUNED_SUMMARY)
    result = {}
    for _, row in df.iterrows():
        try:
            bare = ast.literal_eval(str(row['bare_params']))
            shoe = ast.literal_eval(str(row['shoe_params']))
        except Exception:
            bare = shoe = DEFAULT_PARAMS
        result[row['patient_id']] = {'bare': bare, 'shoe': shoe}
    return result


def get_params(tuned_map, patient_id, trial_name):
    if patient_id in tuned_map:
        return tuned_map[patient_id]['bare'] if trial_name in BARE_TRIALS else tuned_map[patient_id]['shoe']
    return DEFAULT_PARAMS


def load_control_data(patient_id):
    """Load trials from converted CSVs in data/processed/controle/."""
    csv_dir = PROCESSED_DIR / patient_id
    trials  = [apply_sensor_reorder(load_trial_from_csv(csv_dir / f'{t}.csv', t))
               for t in TRIAL_NAMES]
    return {t['Name']: t for t in trials}


def load_control_indices(patient_id):
    """Load index data from CSV (data/processed/controle/) or fall back to .mat."""
    csv_path = PROCESSED_DIR / patient_id / 'indices.csv'
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        indices = {}
        for trial in TRIAL_NAMES:
            tdf = df[df['trial'] == trial]
            if len(tdf) == 0:
                indices[trial] = None
                continue
            row = tdf[tdf['section'] == 0].iloc[0]
            indices[trial] = {
                'indexStart': int(row['index_start']),
                'indexEnd':   int(row['index_end']),
            }
        return indices

    # fallback: load directly from .mat files in Kinematics/Dataset/controle/
    subject_dir = CONTROLE_DIR / patient_id
    indices = {}
    for trial in TRIAL_NAMES:
        idx_file = subject_dir / f'{trial}_indexData.mat'
        if idx_file.exists():
            mat = sio.loadmat(str(idx_file), simplify_cells=True)
            indices[trial] = mat
        else:
            indices[trial] = None
    return indices


def process_one_trial(patient_id, trial_name, trials_dict, indices_dict, force, params=None):
    out_dir   = OUT_ROOT / patient_id / trial_name
    all_exist = all((out_dir / f).exists() for f in [
        'summary.csv', 'events.csv', 'angles_time_series.csv',
        'angles_per_stride_raw.csv', 'angles_per_stride_normalized.csv'
    ])
    if all_exist and not force:
        return 'skip'

    cal = TRIAL_TO_CAL[trial_name]
    td  = trials_dict.get(trial_name)
    cd  = trials_dict.get(cal)
    ti  = indices_dict.get(trial_name)
    ci  = indices_dict.get(cal)
    if any(v is None for v in [td, cd, ti, ci]):
        return 'missing_data'

    swing_params = params if params is not None else DEFAULT_PARAMS
    try:
        result = process_gait_data(cd, ci, td, ti,
                                   swing_params=swing_params, return_debug=True)
    except Exception as e:
        return f'pipeline_failed:{e}'

    try:
        export_trial_csvs(result, patient_id, 'S1', trial_name, out_dir)
        return 'ok'
    except Exception as e:
        return f'export_failed:{e}'


def main():
    force = '--force' in sys.argv
    args  = [a for a in sys.argv[1:] if a != '--force']

    patients = args if args else sorted(
        p.name for p in PROCESSED_DIR.iterdir() if p.is_dir()
    )

    tuned_map = load_tuned_params()
    if tuned_map:
        print(f'  Using tuned params for {len(tuned_map)} subjects')
    else:
        print('  No tuned params found, using defaults')

    total = skipped = done = failed = 0

    print('Exporting joint-angle CSVs — control group')
    print(f'  Subjects : {len(patients)}')
    print(f'  Output   : {OUT_ROOT}\n')

    for patient_id in patients:
        try:
            trials_dict  = load_control_data(patient_id)
            indices_dict = load_control_indices(patient_id)
        except Exception as e:
            print(f'  {patient_id}: load failed — {e}')
            continue

        for trial_name in WALKING_TRIALS:
            total += 1
            params = get_params(tuned_map, patient_id, trial_name)
            status = process_one_trial(
                patient_id, trial_name, trials_dict, indices_dict, force, params
            )

            if status == 'skip':
                skipped += 1
            elif status == 'ok':
                done += 1
                print(f'  {patient_id}/{trial_name}: OK')
            else:
                failed += 1
                print(f'  {patient_id}/{trial_name}: {status}')

    print(f'\n{"="*55}')
    print(f'Total   : {total}')
    print(f'Skipped : {skipped}')
    print(f'Done    : {done}')
    print(f'Failed  : {failed}')
    print(f'Output  : {OUT_ROOT}')
    print(f'{"="*55}')

    # Merge all per-trial CSVs into consolidated group-level files
    merge_exports(OUT_ROOT, MERGE_ROOT)


def merge_exports(src_root, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    names = ['summary', 'events', 'angles_time_series',
             'angles_per_stride_raw', 'angles_per_stride_normalized']
    dfs = {n: [] for n in names}
    for csv_file in src_root.rglob('summary.csv'):
        trial_dir = csv_file.parent
        for name in names:
            f = trial_dir / f'{name}.csv'
            if f.exists() and f.stat().st_size > 0:
                try:
                    dfs[name].append(pd.read_csv(f))
                except Exception:
                    pass
    print('\nMerging into consolidated files...')
    for name, frames in dfs.items():
        if not frames:
            continue
        merged = pd.concat(frames, ignore_index=True)
        merged.to_csv(out_dir / f'{name}.csv', index=False)
        print(f'  {name}.csv: {merged.shape}')
    print(f'  Saved to {out_dir}')


if __name__ == '__main__':
    main()
