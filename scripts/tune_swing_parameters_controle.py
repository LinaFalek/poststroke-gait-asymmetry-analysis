"""
Tune swing-detection parameters for the control group.

Uses Nelder-Mead optimization against reference gait parameters,
identical approach to tune_swing_parameters.py for hemiparetics.

Input:
    data/processed/controle/{patient_id}/          (converted CSVs)
    Kinematics/Dataset/results/controle/{patient_id}/{patient_id}_Result.csv

Output:
    output/controle_tuned/fine_tuning_summary.csv

Usage:
    python scripts/tune_swing_parameters_controle.py                # all control subjects
    python scripts/tune_swing_parameters_controle.py 01-T-CC        # specific subject
"""

import sys
import ast
import warnings
import multiprocessing
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.data_loader import apply_sensor_reorder, TRIAL_NAMES, load_trial_from_csv
from src.gait_processing import process_gait_data
from src.gait_functions import remove_outliers_and_compute_mean
import scipy.io as sio

PROJECT_ROOT   = Path(__file__).parent.parent
PROCESSED_DIR  = PROJECT_ROOT / 'data' / 'processed' / 'controle'
REFERENCE_RESULTS = PROJECT_ROOT / 'Kinematics' / 'Dataset' / 'results' / 'controle'
OUTPUT_DIR     = PROJECT_ROOT / 'output' / 'controle_tuned'

BARE_TRIALS  = ['Bare_fast', 'Bare_pref']
SHOE_TRIALS  = ['Shoe_fast', 'Shoe_pref']
TRIAL_TO_CAL = {
    'Bare_fast': 'Bare_calibration', 'Bare_pref': 'Bare_calibration',
    'Shoe_fast': 'Shoe_calibration', 'Shoe_pref': 'Shoe_calibration',
}

PARAM_BOUNDS    = [(50, 200), (10, 80), (-500, -100), (15, 80), (5, 30)]
DEFAULT_PARAMS  = [100, 30, -300, 30, 10]
N_RANDOM_STARTS = 1

GLOBAL_REFS: dict = {}


def _normalize_param_name(s: str) -> str:
    return (s.replace('maxData st', 'maxData_st')
             .replace('maxData sw', 'maxData_sw')
             .replace('minData st', 'minData_st')
             .replace('minData sw', 'minData_sw')
             .replace('stepLength', 'strideLength'))


def compute_global_refs() -> dict:
    rows = []
    for patient_dir in REFERENCE_RESULTS.iterdir():
        if not patient_dir.is_dir():
            continue
        csv = patient_dir / f'{patient_dir.name}_Result.csv'
        if csv.exists():
            try:
                rows.append(pd.read_csv(csv))
            except Exception:
                pass
    if not rows:
        return {}
    combined = pd.concat(rows, ignore_index=True)
    combined['Parameter'] = combined['Parameter'].apply(_normalize_param_name)
    return (combined.groupby('Parameter')['Mean']
                    .apply(lambda x: x.abs().mean())
                    .to_dict())


def clip_to_bounds(x):
    return np.array([np.clip(x[i], PARAM_BOUNDS[i][0], PARAM_BOUNDS[i][1])
                     for i in range(5)])


def load_control_trials(patient_id):
    csv_dir = PROCESSED_DIR / patient_id
    trials  = [apply_sensor_reorder(load_trial_from_csv(csv_dir / f'{t}.csv', t))
               for t in TRIAL_NAMES]
    trials_dict = {t['Name']: t for t in trials}

    indices_csv = csv_dir / 'indices.csv'
    df = pd.read_csv(indices_csv)
    indices_dict = {}
    for trial in TRIAL_NAMES:
        tdf = df[df['trial'] == trial]
        if len(tdf) == 0:
            indices_dict[trial] = None
            continue
        row = tdf[tdf['section'] == 0].iloc[0]
        indices_dict[trial] = {
            'indexStart': int(row['index_start']),
            'indexEnd':   int(row['index_end']),
        }
    return trials_dict, indices_dict


def run_pipeline(trials_dict, indices_dict, params, trial_filter):
    rows = []
    for trial_name in trial_filter:
        cal = TRIAL_TO_CAL[trial_name]
        td  = trials_dict.get(trial_name)
        cd  = trials_dict.get(cal)
        ti  = indices_dict.get(trial_name)
        ci  = indices_dict.get(cal)
        if any(v is None for v in [td, cd, ti, ci]):
            continue
        try:
            result = process_gait_data(cd, ci, td, ti, swing_params=params)
        except Exception:
            continue
        gp = result['GaitParm']
        gc = result['GaitChar']
        rows.append({'File': trial_name, 'Parameter': 'WalkingSpeed',
                     'Mean': gp['walkingSpeed'], 'Std': 0, 'Unit': 'm/s'})
        for side_key, side_label in [('af', 'Affected'), ('nf', 'Non-paretic')]:
            for pname in ['strideLength', 'strideTime', 'stanceTime',
                          'swingTime', 'stanceDuration']:
                rows.append({'File': trial_name,
                             'Parameter': f'{side_label} {pname}',
                             'Mean': gp[side_key][pname]['meanValue'],
                             'Std':  gp[side_key][pname]['stdValue'],
                             'Unit': 'm' if 'Length' in pname else 's'})
        for jk, jl in [('aa','Ankle'), ('ak','Knee'), ('ah','Hip')]:
            for m in ['maxData_st', 'maxData_sw', 'minData_st', 'minData_sw']:
                _, mv, sv = remove_outliers_and_compute_mean(gc[jk][m])
                rows.append({'File': trial_name,
                             'Parameter': f'Affected {jl} {m}',
                             'Mean': mv, 'Std': sv, 'Unit': 'deg'})
        for jk, jl in [('na','Ankle'), ('nk','Knee'), ('nh','Hip')]:
            for m in ['maxData_st', 'maxData_sw', 'minData_st', 'minData_sw']:
                _, mv, sv = remove_outliers_and_compute_mean(gc[jk][m])
                rows.append({'File': trial_name,
                             'Parameter': f'Non-paretic {jl} {m}',
                             'Mean': mv, 'Std': sv, 'Unit': 'deg'})
    return pd.DataFrame(rows) if rows else None


def compute_error(py_df, ref_df):
    if py_df is None or ref_df is None or len(py_df) == 0:
        return None
    mdf = ref_df.copy()
    mdf['Parameter'] = (mdf['Parameter']
                        .str.replace('maxData st',  'maxData_st',  regex=False)
                        .str.replace('maxData sw',  'maxData_sw',  regex=False)
                        .str.replace('minData st',  'minData_st',  regex=False)
                        .str.replace('minData sw',  'minData_sw',  regex=False)
                        .str.replace('stepLength',  'strideLength', regex=False))
    merged = py_df.merge(mdf, on=['File', 'Parameter'],
                         suffixes=('_py', '_mat'), how='inner')
    if len(merged) == 0:
        return None
    param_ref = merged['Parameter'].map(GLOBAL_REFS)
    fallback  = merged.groupby('Parameter')['Mean_mat'].transform(
                    lambda x: x.abs().mean())
    param_ref = param_ref.fillna(fallback).clip(lower=1e-6)
    merged['rel_diff'] = (np.abs(merged['Mean_py'] - merged['Mean_mat'])
                          / param_ref * 100)
    return merged['rel_diff'].mean()


def tune_condition(trials_dict, indices_dict, ref_df, trial_filter):
    n_evals = [0]

    def objective(x):
        n_evals[0] += 1
        xc     = clip_to_bounds(x)
        params = [int(round(v)) for v in xc]
        py_df  = run_pipeline(trials_dict, indices_dict, params, trial_filter)
        err    = compute_error(py_df, ref_df)
        return err if err is not None else 1e6

    rng    = np.random.default_rng(42)
    starts = [np.array(DEFAULT_PARAMS, dtype=float)]
    for _ in range(N_RANDOM_STARTS):
        starts.append(np.array([rng.uniform(lo, hi) for lo, hi in PARAM_BOUNDS]))

    best_params = DEFAULT_PARAMS
    best_error  = None

    for x0 in starts:
        x0 = clip_to_bounds(x0)
        try:
            res    = minimize(objective, x0, method='Nelder-Mead',
                              options={'maxiter': 400, 'xatol': 1.0,
                                       'fatol': 0.05, 'disp': False})
            params = [int(round(clip_to_bounds(res.x)[j])) for j in range(5)]
            err    = objective(res.x)
            if best_error is None or err < best_error:
                best_error  = err
                best_params = params
        except Exception:
            pass

    return best_params, best_error, n_evals[0]


def _worker(patient_id):
    log = []
    try:
        trials_dict, indices_dict = load_control_trials(patient_id)
    except Exception as e:
        log.append(f'  SKIP - load failed: {e}')
        return patient_id, None, log

    ref_path = REFERENCE_RESULTS / patient_id / f'{patient_id}_Result.csv'
    if not ref_path.exists():
        log.append(f'  SKIP - no reference data')
        return patient_id, None, log

    ref_df = pd.read_csv(ref_path)
    results   = {}

    for cond, trial_filter in [('bare', BARE_TRIALS), ('shoe', SHOE_TRIALS)]:
        ref_trials = set(ref_df['File'].unique()) if 'File' in ref_df.columns else set()
        has_reference    = any(t in ref_trials for t in trial_filter)
        has_data      = any(trials_dict.get(t) is not None for t in trial_filter)

        if not has_reference or not has_data:
            log.append(f'  {cond}: no data/reference')
            results[f'{cond}_params'] = DEFAULT_PARAMS
            results[f'{cond}_error']  = None
            results[f'{cond}_status'] = 'no_data'
            continue

        params, error, n_ev = tune_condition(
            trials_dict, indices_dict, ref_df, trial_filter)

        log.append(f'  {cond}: {params}  error={error:.2f}%  evals={n_ev}')
        results[f'{cond}_params'] = params
        results[f'{cond}_error']  = round(error, 4) if error else None
        results[f'{cond}_status'] = 'ok'

    # Pick single best params (lower error between bare/shoe)
    bare_err = results.get('bare_error') or 1e6
    shoe_err = results.get('shoe_error') or 1e6
    fine_params = results['bare_params'] if bare_err <= shoe_err else results['shoe_params']
    fine_error  = min(bare_err, shoe_err) if min(bare_err, shoe_err) < 1e6 else None

    results['fine_params'] = fine_params
    results['fine_error']  = fine_error
    return patient_id, results, log


def main():
    global GLOBAL_REFS
    cli_patients = sys.argv[1:]

    print('Computing global parameter reference scales...')
    GLOBAL_REFS = compute_global_refs()
    print(f'  {len(GLOBAL_REFS)} parameters indexed.\n')

    if cli_patients:
        patients = cli_patients
    else:
        patients = sorted(p.name for p in PROCESSED_DIR.iterdir() if p.is_dir())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    n_workers = min(multiprocessing.cpu_count(), len(patients))
    print(f'Tuning {len(patients)} control subjects using {n_workers} workers')
    print(f'Output: {OUTPUT_DIR}\n')

    rows = []
    done = 0

    with multiprocessing.Pool(processes=n_workers) as pool:
        for patient_id, res, log in pool.imap_unordered(_worker, patients):
            done += 1
            print(f'[{done}/{len(patients)}] {patient_id}')
            for line in log:
                print(line)
            if res is None:
                continue
            rows.append({
                'patient_id':   patient_id,
                'fine_params':  str(res['fine_params']),
                'fine_error_%': res.get('fine_error'),
                'bare_params':  str(res.get('bare_params')),
                'bare_error_%': res.get('bare_error'),
                'shoe_params':  str(res.get('shoe_params')),
                'shoe_error_%': res.get('shoe_error'),
            })

    out_df   = pd.DataFrame(rows).sort_values('patient_id')
    out_path = OUTPUT_DIR / 'fine_tuning_summary.csv'
    out_df.to_csv(out_path, index=False)

    print(f'\n{"="*60}')
    print(f'{"Patient":<12}  {"Fine params":<28}  {"Err%":>6}')
    print('-' * 52)
    for _, r in out_df.iterrows():
        fe = f'{r["fine_error_%"]:.1f}' if pd.notna(r.get('fine_error_%')) else 'N/A'
        print(f'{r["patient_id"]:<12}  {str(r["fine_params"]):<28}  {fe:>6}')
    print(f'\n[OK] Saved -> {out_path}')


if __name__ == '__main__':
    main()
