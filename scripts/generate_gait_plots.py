"""
Batch Gait Plots - All Patients / All RDVs / All Trials
=========================================================

Folder structure:
    output/figures/all_patients/{patient_id}/{RDV}/{trial}/
        01_joint_angles.png       Hip/Knee/Ankle gait cycles, Affected vs Non-paretic
        02_foot_acceleration.png  Foot acceleration norm + swing detection phases
        03_spatiotemporal.png     Spatiotemporal parameters for this trial

Usage:
    python scripts/plot_all_patients.py                        # all
    python scripts/plot_all_patients.py 01-P-AR                # one patient
    python scripts/plot_all_patients.py 01-P-AR RDV1           # one patient + RDV
    python scripts/plot_all_patients.py 01-P-AR RDV1 Bare_pref # one specific
    python scripts/plot_all_patients.py --force                 # regenerate all

Skips figures that already exist (safe to re-run).
"""

import sys
import ast
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings('ignore')   # suppress numpy std warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.data_loader import load_patient_data, load_index_files
from src.gait_processing import process_gait_data

PROJECT_ROOT = Path(__file__).parent.parent
TUNED_DIR    = PROJECT_ROOT / 'output' / 'hemiparetic_tuned'
LONG_DIR     = PROJECT_ROOT / 'output' / 'longitudinal'
OUT_ROOT     = PROJECT_ROOT / 'output' / 'figures' / 'all_patients'

RDVS = ['RDV1', 'RDV2', 'RDV3', 'RDV4', 'RDV5']
WALKING_TRIALS = ['Bare_fast', 'Bare_pref', 'Shoe_fast', 'Shoe_pref']
TRIAL_TO_CAL   = {
    'Bare_fast': 'Bare_calibration', 'Bare_pref': 'Bare_calibration',
    'Shoe_fast': 'Shoe_calibration', 'Shoe_pref': 'Shoe_calibration',
}

COL_AF = '#c0392b'   # red — affected
COL_NF = '#2980b9'   # blue — non-paretic

BARE_TRIALS = {'Bare_fast', 'Bare_pref'}
SHOE_TRIALS = {'Shoe_fast', 'Shoe_pref'}
CONDITION_PARAMS_PATH = TUNED_DIR / 'condition_params.csv'


# ── data helpers ─────────────────────────────────────────────────────────────

def load_all_tuned_params():
    """Load per-condition params if available, else fall back to fine_params."""
    # Always load the single-condition fallback
    df_fine = pd.read_csv(TUNED_DIR / 'fine_tuning_summary.csv')
    fine = {row['patient_id']: ast.literal_eval(row['fine_params'])
            for _, row in df_fine.iterrows()}

    # Try to load per-condition params
    if not CONDITION_PARAMS_PATH.exists():
        # Return (fine, fine) for bare/shoe — same fallback for both
        return {pid: {'bare': p, 'shoe': p, 'fallback': p}
                for pid, p in fine.items()}

    df_cond = pd.read_csv(CONDITION_PARAMS_PATH)
    result = {}
    for _, row in df_cond.iterrows():
        pid = row['patient_id']
        fb  = fine.get(pid)

        def _parse(val):
            try:
                return ast.literal_eval(str(val))
            except Exception:
                return fb

        bare_p = _parse(row.get('bare_params'))
        shoe_p = _parse(row.get('shoe_params'))
        result[pid] = {
            'bare':     bare_p if bare_p is not None else fb,
            'shoe':     shoe_p if shoe_p is not None else fb,
            'fallback': fb,
        }
    # For patients not in condition_params.csv, fall back to fine
    for pid, p in fine.items():
        if pid not in result:
            result[pid] = {'bare': p, 'shoe': p, 'fallback': p}
    return result


def get_trial_params(params_entry, trial_name):
    """Pick bare or shoe params based on trial name."""
    if isinstance(params_entry, dict):
        if trial_name in BARE_TRIALS:
            return params_entry['bare']
        return params_entry['shoe']
    return params_entry  # legacy plain list


def load_ok_pairs():
    log = pd.read_csv(LONG_DIR / 'processing_log.csv')
    return set(zip(log[log['status'] == 'ok']['patient_id'],
                   log[log['status'] == 'ok']['rdv']))


def run_trial(trials, indices, trial_name, params):
    cal = TRIAL_TO_CAL[trial_name]
    td, cd = trials.get(trial_name), trials.get(cal)
    ti, ci = indices.get(trial_name), indices.get(cal)
    if any(v is None for v in [td, cd, ti, ci]):
        return None
    try:
        return process_gait_data(cd, ci, td, ti,
                                 swing_params=params, return_debug=True)
    except Exception:
        return None


def iqr_filter(data):
    """Remove outlier strides using IQR on per-stride mean."""
    if data.shape[0] == 0:
        return data
    sm = data.mean(axis=1)
    q1, q3 = np.percentile(sm, [25, 75])
    iqr = q3 - q1
    mask = (sm >= q1 - 2*iqr) & (sm <= q3 + 2*iqr)
    return data[mask]


# ── Figure 1: Joint angle gait cycles ────────────────────────────────────────

def plot_joint_angles(result, patient_id, rdv, trial_name, params, out_path):
    """
    3 rows x 2 cols: Hip / Knee / Ankle  x  Affected / Non-paretic
    Mean +/- std band over IQR-filtered strides.
    """
    cd = result['CycleD']
    T  = cd['T']
    gp = result['GaitParm']

    joint_keys = [('ah', 'nh', 'Hip'),
                  ('ak', 'nk', 'Knee'),
                  ('aa', 'na', 'Ankle')]

    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    fig.suptitle(
        f'Joint Angle Gait Cycles\n'
        f'{patient_id}  |  {rdv}  |  {trial_name}  |  params: {params}',
        fontsize=11, fontweight='bold'
    )

    for row, (jaf, jnf, jlabel) in enumerate(joint_keys):
        for col, (jkey, side, color, side_key) in enumerate([
            (jaf, 'Affected',    COL_AF, 'af'),
            (jnf, 'Non-paretic', COL_NF, 'nf'),
        ]):
            ax = axes[row, col]
            data = iqr_filter(cd[jkey])
            st   = gp[side_key]['stanceDuration']['meanValue']

            if data.shape[0] > 0:
                mean_c = data.mean(axis=0)
                std_c  = data.std(axis=0)

                for stride in data:
                    ax.plot(T, stride, color=color, alpha=0.08, linewidth=0.5)

                ax.fill_between(T, mean_c - std_c, mean_c + std_c,
                                color=color, alpha=0.2)
                ax.plot(T, mean_c, color=color, linewidth=2.2,
                        label=f'{side}  n={data.shape[0]} strides')

                ax.axvline(st, color='grey', linestyle='--',
                           linewidth=1.0, alpha=0.7, label=f'Stance end ({st:.1f}%)')

            ax.axhline(0, color='black', linewidth=0.5, linestyle=':')
            ax.set_title(f'{jlabel}  —  {side}', fontsize=10)
            ax.set_ylabel(f'{jlabel} angle (deg)' if col == 0 else '')
            ax.set_xlabel('Gait cycle (%)' if row == 2 else '')
            ax.set_xlim(0, 100)
            ax.legend(fontsize=8, loc='upper right', framealpha=0.7)
            ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# ── Figure 2: Foot acceleration ("force") + swing detection ──────────────────

def plot_foot_acceleration(result, patient_id, rdv, trial_name, params, out_path):
    """
    4 stacked subplots:
      - Affected foot |acc| norm  (IMU proxy for ground contact force)
      - Affected swing signal
      - Non-paretic foot |acc| norm
      - Non-paretic swing signal
    First 20 s of signal shown.
    """
    dbg    = result['_debug']
    swing  = dbg['swing']
    fa_acc = dbg['foot_acc']
    t_vec  = dbg['time_vec'] / 1000.0   # ms -> s

    WIN = min(len(t_vec), 2000)   # ~20 s @ 100 Hz
    t   = t_vec[:WIN]

    def norm2d(arr):
        a = arr[:WIN]
        return np.linalg.norm(a, axis=1) if a.ndim == 2 else a

    af_acc = norm2d(fa_acc['af'])
    nf_acc = norm2d(fa_acc['nf'])
    sw_af  = swing['af'][:WIN]
    sw_nf  = swing['nf'][:WIN]

    def shade(ax, sw, color, alpha=0.25):
        in_sw, start = False, 0
        for i, v in enumerate(sw):
            if v == 1 and not in_sw:
                in_sw, start = True, i
            elif v == 0 and in_sw:
                ax.axvspan(t[start], t[i], color=color, alpha=alpha)
                in_sw = False
        if in_sw:
            ax.axvspan(t[start], t[-1], color=color, alpha=alpha)

    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    fig.suptitle(
        f'Foot Acceleration + Swing Detection\n'
        f'{patient_id}  |  {rdv}  |  {trial_name}  |  params: {params}',
        fontsize=11, fontweight='bold'
    )

    # --- Affected foot acceleration ---
    ax = axes[0]
    ax.plot(t, af_acc, color=COL_AF, linewidth=0.8)
    shade(ax, sw_af, COL_AF)
    ax.set_ylabel('|acc| (m/s²)')
    ax.set_title('Affected foot — acceleration norm  (shading = swing phase)')
    ax.grid(alpha=0.3)

    # --- Affected swing signal ---
    ax = axes[1]
    ax.fill_between(t, sw_af.astype(float), step='post',
                    color=COL_AF, alpha=0.6)
    ax.set_ylim(-0.1, 1.3)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Stance', 'Swing'])
    ax.set_title('Affected leg — swing detection signal')
    ax.grid(alpha=0.3)

    # --- Non-paretic foot acceleration ---
    ax = axes[2]
    ax.plot(t, nf_acc, color=COL_NF, linewidth=0.8)
    shade(ax, sw_nf, COL_NF)
    ax.set_ylabel('|acc| (m/s²)')
    ax.set_title('Non-paretic foot — acceleration norm  (shading = swing phase)')
    ax.grid(alpha=0.3)

    # --- Non-paretic swing signal ---
    ax = axes[3]
    ax.fill_between(t, sw_nf.astype(float), step='post',
                    color=COL_NF, alpha=0.6)
    ax.set_ylim(-0.1, 1.3)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Stance', 'Swing'])
    ax.set_title('Non-paretic leg — swing detection signal')
    ax.set_xlabel('Time (s)')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# ── Figure 3: Spatiotemporal parameters ──────────────────────────────────────

def plot_spatiotemporal(result, patient_id, rdv, trial_name, params, out_path):
    """
    2 x 3 grid of bars: one value per parameter, Affected vs Non-paretic.
    """
    gp = result['GaitParm']

    param_defs = [
        ('walkingSpeed',   'Walking Speed (m/s)'),
        ('strideLength',   'Stride Length (m)'),
        ('strideTime',     'Stride Time (s)'),
        ('stanceTime',     'Stance Time (s)'),
        ('swingTime',      'Swing Time (s)'),
        ('stanceDuration', 'Stance Duration (%)'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle(
        f'Spatiotemporal Parameters\n'
        f'{patient_id}  |  {rdv}  |  {trial_name}  |  params: {params}',
        fontsize=11, fontweight='bold'
    )
    axes = axes.flatten()

    labels = ['Affected', 'Non-paretic']
    colors = [COL_AF, COL_NF]

    for idx, (pname, ylabel) in enumerate(param_defs):
        ax = axes[idx]

        if pname == 'walkingSpeed':
            val = gp['walkingSpeed']
            ax.bar(['Overall'], [val], color='#7f8c8d', edgecolor='white', width=0.4)
            ax.set_ylim(bottom=0)
        else:
            vals = []
            stds = []
            for side in ['af', 'nf']:
                v = gp[side][pname]
                vals.append(v['meanValue'] if isinstance(v, dict) else v)
                stds.append(v['stdValue']  if isinstance(v, dict) else 0)

            x = np.arange(2)
            bars = ax.bar(x, vals, yerr=stds, color=colors,
                          capsize=5, edgecolor='white',
                          error_kw={'elinewidth': 1.2})
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_ylim(bottom=0)

            # Value labels on bars
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() * 1.02,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=8)

        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(ylabel.split('(')[0].strip(), fontsize=10, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    # Asymmetry note
    fig.text(0.5, 0.01,
             'Red = Affected side   |   Blue = Non-paretic side   |   Error bars = 1 std',
             ha='center', fontsize=9, color='#555')

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# ── per-trial processing ──────────────────────────────────────────────────────

def process_one_trial(patient_id, rdv, trial_name, trials, indices, params, force=False):
    """
    Generate 3 figures for one patient/RDV/trial.
    Returns status string.
    """
    trial_dir = OUT_ROOT / patient_id / rdv / trial_name
    trial_dir.mkdir(parents=True, exist_ok=True)

    p_ang  = trial_dir / '01_joint_angles.png'
    p_acc  = trial_dir / '02_foot_acceleration.png'
    p_spat = trial_dir / '03_spatiotemporal.png'

    all_exist = p_ang.exists() and p_acc.exists() and p_spat.exists()
    if all_exist and not force:
        return 'skip'

    result = run_trial(trials, indices, trial_name, params)
    if result is None:
        return 'pipeline_failed'

    try:
        if not p_ang.exists() or force:
            plot_joint_angles(result, patient_id, rdv, trial_name, params, p_ang)
        if not p_acc.exists() or force:
            plot_foot_acceleration(result, patient_id, rdv, trial_name, params, p_acc)
        if not p_spat.exists() or force:
            plot_spatiotemporal(result, patient_id, rdv, trial_name, params, p_spat)
        return 'ok'
    except Exception as e:
        return f'err:{e}'


# ── main batch loop ───────────────────────────────────────────────────────────

def main():
    force = '--force' in sys.argv
    args  = [a for a in sys.argv[1:] if a != '--force']

    cli_patients = [a for a in args if not a.startswith('RDV') and '_' not in a]
    cli_rdvs     = [a for a in args if a.startswith('RDV')]
    cli_trials   = [a for a in args if '_' in a]

    params_map = load_all_tuned_params()
    ok_pairs   = load_ok_pairs()

    patients = cli_patients if cli_patients else sorted(params_map.keys())
    rdvs     = cli_rdvs     if cli_rdvs     else RDVS
    trials   = cli_trials   if cli_trials   else WALKING_TRIALS

    total = skipped = done = failed = 0

    print(f'Generating gait plots...')
    print(f'  Patients : {len(patients)}')
    print(f'  RDVs     : {rdvs}')
    print(f'  Trials   : {trials}')
    print(f'  Output   : {OUT_ROOT}\n')

    for patient_id in patients:
        if patient_id not in params_map:
            print(f'  {patient_id}: no tuned params, skip')
            continue

        params_entry = params_map[patient_id]

        for rdv in rdvs:
            if (patient_id, rdv) not in ok_pairs:
                continue

            # Load patient data once per patient/RDV
            try:
                re_output, _ = load_patient_data(patient_id, rdv)
            except FileNotFoundError:
                print(f'  {patient_id}/{rdv}: data not found')
                continue

            indices_dict = load_index_files(patient_id, rdv)
            trials_dict  = {t['Name']: t for t in re_output}

            for trial_name in trials:
                total += 1
                params = get_trial_params(params_entry, trial_name)
                status = process_one_trial(
                    patient_id, rdv, trial_name,
                    trials_dict, indices_dict, params, force=force
                )

                if status == 'skip':
                    skipped += 1
                elif status == 'ok':
                    done += 1
                    print(f'  {patient_id}/{rdv}/{trial_name}: OK')
                elif status == 'pipeline_failed':
                    failed += 1
                    print(f'  {patient_id}/{rdv}/{trial_name}: pipeline failed')
                else:
                    failed += 1
                    print(f'  {patient_id}/{rdv}/{trial_name}: {status}')

    print(f'\n{"="*55}')
    print(f'Total trial/RDV combos : {total}')
    print(f'Already done (skipped) : {skipped}')
    print(f'Newly generated        : {done}')
    print(f'Failed                 : {failed}')
    print(f'Output                 : {OUT_ROOT}')
    print(f'{"="*55}')


if __name__ == '__main__':
    main()
