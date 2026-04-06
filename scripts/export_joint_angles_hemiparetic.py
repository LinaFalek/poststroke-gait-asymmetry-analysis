"""
Export joint-angle CSV outputs for the hemiparetic group.

For each patient / RDV / trial, runs the gait pipeline and writes:
    output/csv_exports/hemiparetic/{patient_id}/{rdv}/{trial_name}/
        summary.csv
        events.csv
        angles_time_series.csv
        angles_per_stride_raw.csv
        angles_per_stride_normalized.csv

Usage:
    python scripts/export_joint_angles_hemiparetic.py                   # all patients
    python scripts/export_joint_angles_hemiparetic.py 01-P-AR           # one patient
    python scripts/export_joint_angles_hemiparetic.py 01-P-AR RDV1 RDV2 # specific RDVs
    python scripts/export_joint_angles_hemiparetic.py --force           # overwrite existing
"""

import sys
import ast
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings('ignore')

import pandas as pd

from src.data_loader import load_patient_data, load_index_files
from src.gait_processing import process_gait_data
from src.csv_exporter import export_trial_csvs

PROJECT_ROOT = Path(__file__).parent.parent
TUNED_DIR    = PROJECT_ROOT / 'output' / 'hemiparetic_tuned'
LONG_DIR     = PROJECT_ROOT / 'output' / 'longitudinal'
OUT_ROOT     = PROJECT_ROOT / 'output' / 'csv_exports' / 'hemiparetic'

RDVS           = ['RDV1', 'RDV2', 'RDV3', 'RDV4', 'RDV5']
WALKING_TRIALS = ['Bare_fast', 'Bare_pref', 'Shoe_fast', 'Shoe_pref']
TRIAL_TO_CAL   = {
    'Bare_fast': 'Bare_calibration', 'Bare_pref': 'Bare_calibration',
    'Shoe_fast': 'Shoe_calibration', 'Shoe_pref': 'Shoe_calibration',
}
BARE_TRIALS           = {'Bare_fast', 'Bare_pref'}
CONDITION_PARAMS_PATH = TUNED_DIR / 'condition_params.csv'


def load_all_tuned_params():
    df_fine = pd.read_csv(TUNED_DIR / 'fine_tuning_summary.csv')
    fine = {row['patient_id']: ast.literal_eval(row['fine_params'])
            for _, row in df_fine.iterrows()}

    if not CONDITION_PARAMS_PATH.exists():
        return {pid: {'bare': p, 'shoe': p} for pid, p in fine.items()}

    df_cond = pd.read_csv(CONDITION_PARAMS_PATH)
    result  = {}
    for _, row in df_cond.iterrows():
        pid = row['patient_id']
        fb  = fine.get(pid)
        def _parse(val):
            try:
                return ast.literal_eval(str(val))
            except Exception:
                return fb
        result[pid] = {
            'bare': _parse(row.get('bare_params')) or fb,
            'shoe': _parse(row.get('shoe_params')) or fb,
        }
    for pid, p in fine.items():
        if pid not in result:
            result[pid] = {'bare': p, 'shoe': p}
    return result


def get_trial_params(params_entry, trial_name):
    if isinstance(params_entry, dict):
        return params_entry['bare'] if trial_name in BARE_TRIALS else params_entry['shoe']
    return params_entry


def load_ok_pairs():
    log_path = LONG_DIR / 'processing_log.csv'
    if not log_path.exists():
        return None  # no filter, try all
    log = pd.read_csv(log_path)
    return set(zip(log[log['status'] == 'ok']['patient_id'],
                   log[log['status'] == 'ok']['rdv']))


def process_one_trial(patient_id, rdv, trial_name, trials_dict, indices_dict, params, force):
    out_dir   = OUT_ROOT / patient_id / rdv / trial_name
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

    try:
        result = process_gait_data(cd, ci, td, ti,
                                   swing_params=params, return_debug=True)
    except Exception as e:
        return f'pipeline_failed:{e}'

    try:
        export_trial_csvs(result, patient_id, rdv, trial_name, out_dir)
        return 'ok'
    except Exception as e:
        return f'export_failed:{e}'


def main():
    force = '--force' in sys.argv
    args  = [a for a in sys.argv[1:] if a != '--force']

    cli_patients = [a for a in args if not a.startswith('RDV') and '_' not in a]
    cli_rdvs     = [a for a in args if a.startswith('RDV')]

    params_map = load_all_tuned_params()
    ok_pairs   = load_ok_pairs()

    patients = cli_patients if cli_patients else sorted(params_map.keys())
    rdvs     = cli_rdvs     if cli_rdvs     else RDVS

    total = skipped = done = failed = 0

    print('Exporting joint-angle CSVs — hemiparetic group')
    print(f'  Patients : {len(patients)}')
    print(f'  RDVs     : {rdvs}')
    print(f'  Output   : {OUT_ROOT}\n')

    for patient_id in patients:
        if patient_id not in params_map:
            print(f'  {patient_id}: no tuned params, skip')
            continue

        params_entry = params_map[patient_id]

        for rdv in rdvs:
            if ok_pairs is not None and (patient_id, rdv) not in ok_pairs:
                continue

            try:
                re_output, _ = load_patient_data(patient_id, rdv)
            except FileNotFoundError:
                print(f'  {patient_id}/{rdv}: data not found')
                continue

            indices_dict = load_index_files(patient_id, rdv)
            trials_dict  = {t['Name']: t for t in re_output}

            for trial_name in WALKING_TRIALS:
                total += 1
                params = get_trial_params(params_entry, trial_name)
                status = process_one_trial(
                    patient_id, rdv, trial_name,
                    trials_dict, indices_dict, params, force
                )

                if status == 'skip':
                    skipped += 1
                elif status == 'ok':
                    done += 1
                    print(f'  {patient_id}/{rdv}/{trial_name}: OK')
                else:
                    failed += 1
                    print(f'  {patient_id}/{rdv}/{trial_name}: {status}')

    print(f'\n{"="*55}')
    print(f'Total   : {total}')
    print(f'Skipped : {skipped}')
    print(f'Done    : {done}')
    print(f'Failed  : {failed}')
    print(f'Output  : {OUT_ROOT}')
    print(f'{"="*55}')

    # Merge all per-trial CSVs into consolidated group-level files
    merge_exports(OUT_ROOT, PROJECT_ROOT / 'output' / 'csv_exports_merged' / 'hemiparetic')


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
