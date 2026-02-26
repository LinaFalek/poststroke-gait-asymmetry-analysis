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

from src.data_loader import load_patient_data, load_index_files
from src.gait_processing import process_gait_data
from src.gait_functions import remove_outliers_and_compute_mean

PROJECT_ROOT   = Path(__file__).parent.parent
OUTPUT_DIR     = PROJECT_ROOT / 'output' / 'hemiparetic_tuned'
MATLAB_RESULTS = PROJECT_ROOT / 'Kinematics' / 'Dataset' / 'results' / 'hemiparetic'

BARE_TRIALS = ['Bare_fast', 'Bare_pref']
SHOE_TRIALS = ['Shoe_fast', 'Shoe_pref']
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
    for patient_dir in MATLAB_RESULTS.iterdir():
        if not patient_dir.is_dir():
            continue
        for rdv_dir in patient_dir.iterdir():
            csv = rdv_dir / f'{patient_dir.name}_Result.csv'
            if csv.exists():
                try:
                    rows.append(pd.read_csv(csv))
                except Exception:
                    pass
    if not rows:
        return {}
    combined = pd.concat(rows, ignore_index=True)
    combined['Parameter'] = combined['Parameter'].apply(_normalize_param_name)
    refs = (combined.groupby('Parameter')['Mean']
                    .apply(lambda x: x.abs().mean())
                    .to_dict())
    return refs


def clip_to_bounds(x):
    return np.array([np.clip(x[i], PARAM_BOUNDS[i][0], PARAM_BOUNDS[i][1])
                     for i in range(5)])


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


def compute_error(py_df, matlab_df):
    if py_df is None or matlab_df is None or len(py_df) == 0:
        return None
    mdf = matlab_df.copy()
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


def tune_condition(trials_dict, indices_dict, matlab_df, trial_filter, warm_start=None):
    n_evals = [0]

    def objective(x):
        n_evals[0] += 1
        xc     = clip_to_bounds(x)
        params = [int(round(v)) for v in xc]
        py_df  = run_pipeline(trials_dict, indices_dict, params, trial_filter)
        err    = compute_error(py_df, matlab_df)
        return err if err is not None else 1e6

    rng    = np.random.default_rng(42)
    starts = []
    if warm_start is not None:
        starts.append(np.array(warm_start, dtype=float))
    starts.append(np.array(DEFAULT_PARAMS, dtype=float))
    for _ in range(N_RANDOM_STARTS):
        starts.append(np.array([rng.uniform(lo, hi) for lo, hi in PARAM_BOUNDS]))

    best_params = warm_start if warm_start else DEFAULT_PARAMS
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


def _worker(args):
    patient_id, existing_params = args
    log = []

    try:
        re_output, _ = load_patient_data(patient_id, 'RDV1')
    except FileNotFoundError as e:
        log.append(f'  SKIP - no data: {e}')
        return patient_id, None, log

    indices_dict = load_index_files(patient_id, 'RDV1')
    trials_dict  = {t['Name']: t for t in re_output}

    matlab_path = MATLAB_RESULTS / patient_id / 'RDV1' / f'{patient_id}_Result.csv'
    if not matlab_path.exists():
        log.append(f'  SKIP - no MATLAB reference')
        return patient_id, None, log

    matlab_df = pd.read_csv(matlab_path)
    warm      = existing_params.get(patient_id)
    results   = {}

    for cond, trial_filter in [('bare', BARE_TRIALS), ('shoe', SHOE_TRIALS)]:
        matlab_trials = set(matlab_df['File'].unique()) if 'File' in matlab_df.columns else set()
        has_matlab    = any(t in matlab_trials for t in trial_filter)
        has_data      = any(trials_dict.get(t) is not None for t in trial_filter)

        if not has_matlab or not has_data:
            log.append(f'  {cond}: no data/reference -> using existing params')
            results[f'{cond}_params'] = warm
            results[f'{cond}_error']  = None
            results[f'{cond}_status'] = 'no_data'
            continue

        params, error, n_ev = tune_condition(
            trials_dict, indices_dict, matlab_df, trial_filter, warm_start=warm)

        if warm is not None and error is not None:
            warm_err = compute_error(
                run_pipeline(trials_dict, indices_dict, warm, trial_filter), matlab_df)
            if warm_err is not None and warm_err < error:
                params = warm
                error  = warm_err
                log.append(f'  {cond}: warm start better -> {warm}')

        log.append(f'  {cond}: {params}  error={error:.2f}%  evals={n_ev}')
        results[f'{cond}_params'] = params
        results[f'{cond}_error']  = round(error, 4) if error else None
        results[f'{cond}_status'] = 'ok'

    return patient_id, results, log


def load_existing_params():
    path = OUTPUT_DIR / 'fine_tuning_summary.csv'
    if not path.exists():
        return {}
    df  = pd.read_csv(path)
    out = {}
    for _, row in df.iterrows():
        try:
            out[row['patient_id']] = ast.literal_eval(row['fine_params'])
        except Exception:
            pass
    return out


def main():
    global GLOBAL_REFS
    cli_patients = sys.argv[1:]
    existing     = load_existing_params()

    print('Computing global parameter reference scales from MATLAB data...')
    GLOBAL_REFS = compute_global_refs()
    print(f'  {len(GLOBAL_REFS)} parameters indexed.\n')

    if cli_patients:
        patients = cli_patients
    else:
        processed = PROJECT_ROOT / 'data' / 'processed'
        patients  = sorted(p.name for p in processed.iterdir()
                           if p.is_dir() and (p / 'RDV1').exists()
                           and p.name in existing)

    n_workers = min(multiprocessing.cpu_count(), len(patients))
    print(f'Per-condition tuning for {len(patients)} patients '
          f'using {n_workers} parallel workers')
    print(f'Output: {OUTPUT_DIR / "condition_params.csv"}\n')

    args = [(pid, existing) for pid in patients]

    rows     = []
    done     = 0
    existing_fine = existing

    with multiprocessing.Pool(processes=n_workers) as pool:
        for patient_id, res, log in pool.imap_unordered(_worker, args):
            done += 1
            print(f'[{done}/{len(patients)}] {patient_id}')
            for line in log:
                print(line)
            if res is None:
                continue
            rows.append({
                'patient_id':    patient_id,
                'bare_params':   str(res.get('bare_params')),
                'bare_error_%':  res.get('bare_error'),
                'bare_status':   res.get('bare_status', 'ok'),
                'shoe_params':   str(res.get('shoe_params')),
                'shoe_error_%':  res.get('shoe_error'),
                'shoe_status':   res.get('shoe_status', 'ok'),
                'fallback_params': str(existing_fine.get(patient_id)),
            })

    out_df   = pd.DataFrame(rows).sort_values('patient_id')
    out_path = OUTPUT_DIR / 'condition_params.csv'
    out_df.to_csv(out_path, index=False)

    print(f'\n{"="*60}')
    print(f'{"Patient":<12}  {"Bare params":<28}  {"Err%":>6}  '
          f'{"Shoe params":<28}  {"Err%":>6}')
    print('-' * 88)
    for _, r in out_df.iterrows():
        be = f'{r["bare_error_%"]:.1f}' if pd.notna(r.get('bare_error_%')) else 'N/A'
        se = f'{r["shoe_error_%"]:.1f}' if pd.notna(r.get('shoe_error_%')) else 'N/A'
        print(f'{r["patient_id"]:<12}  {str(r["bare_params"]):<28}  {be:>6}  '
              f'{str(r["shoe_params"]):<28}  {se:>6}')

    print(f'\n[OK] Saved -> {out_path}')


if __name__ == '__main__':
    main()
