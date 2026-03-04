import sys, warnings, ast
warnings.filterwarnings('ignore')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from scipy.stats import mannwhitneyu
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from src.data_loader import load_patient_data, load_index_files
from src.gait_processing import process_gait_data
from src.gait_functions import remove_outliers_and_compute_mean

ROOT        = Path(__file__).parent.parent.parent
DATA_DIR    = Path(__file__).parent / 'data'
RESULTS_DIR = Path(__file__).parent / 'results'
FIG_DIR     = Path(__file__).parent / 'figures'
for d in [DATA_DIR, RESULTS_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CONFIG_CSV  = DATA_DIR / 'config.csv'
LABELS_CSV  = ROOT / 'data' / 'patient_labels.csv'
RAW_PSD_CSV = ROOT / 'data' / 'processed' / '01-P-AR' / 'RDV1' / 'Bare_pref.csv'

MIN_STRIDES = 5

MODELS = {
    'Logistic Reg':  LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000, random_state=42),
    'SVM Linear':    SVC(kernel='linear', class_weight='balanced', probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42),
    'Naive Bayes':   GaussianNB(),
}


def load_config():
    df = pd.read_csv(CONFIG_CSV)
    return {row['patient_id']: (row['rdv'], ast.literal_eval(row['params']))
            for _, row in df.iterrows()}


def run_trial(patient_id, rdv, params):
    try:
        re_output, _ = load_patient_data(patient_id, rdv, base_path=ROOT / 'Kinematics')
        idx = load_index_files(patient_id, rdv, base_path=ROOT / 'Kinematics')
        td  = {t['Name']: t for t in re_output}
        result = process_gait_data(
            td['Bare_calibration'], idx['Bare_calibration'],
            td['Bare_pref'],        idx['Bare_pref'],
            swing_params=params)
        n_af = len(result['GaitChar']['ak']['maxData_sw'])
        n_nf = len(result['GaitChar']['nk']['maxData_sw'])
        return result, n_af, n_nf
    except Exception:
        return None, 0, 0


def extract_features(result):
    gp  = result['GaitParm']
    gc  = result['GaitChar']
    row = {'walkingSpeed': gp['walkingSpeed']}
    for side_key, prefix in [('af', 'aff'), ('nf', 'npar')]:
        for param in ['strideLength', 'strideTime', 'stanceTime', 'swingTime', 'stanceDuration']:
            row[f'{prefix}_{param}'] = gp[side_key][param]['meanValue']
    for jk, jl in [('aa', 'Ankle'), ('ak', 'Knee'), ('ah', 'Hip')]:
        for m in ['maxData_st', 'maxData_sw', 'minData_st', 'minData_sw']:
            _, mv, _ = remove_outliers_and_compute_mean(gc[jk][m])
            row[f'aff_{jl}_{m}'] = mv
    for jk, jl in [('na', 'Ankle'), ('nk', 'Knee'), ('nh', 'Hip')]:
        for m in ['maxData_st', 'maxData_sw', 'minData_st', 'minData_sw']:
            _, mv, _ = remove_outliers_and_compute_mean(gc[jk][m])
            row[f'npar_{jl}_{m}'] = mv
    return row


def build_feature_matrix(config, label_map):
    rows, stride_rows = [], []
    for pid, (rdv, params) in config.items():
        result, n_af, n_nf = run_trial(pid, rdv, params)
        stride_rows.append({
            'patient_id': pid, 'rdv': rdv, 'params': str(params),
            'n_af': n_af, 'n_nf': n_nf,
            'paretic_side': label_map.get(pid, 'unknown'),
        })
        if result is None:
            print(f'  SKIP {pid}: pipeline failed')
            continue
        feats = extract_features(result)
        feats.update({'patient_id': pid, 'paretic_side': label_map.get(pid, 'unknown'),
                      'rdv': rdv, 'n_af': n_af, 'n_nf': n_nf, 'params_used': str(params)})
        rows.append(feats)
        print(f'  {pid} ({rdv}): paretic={feats["paretic_side"]}  af={n_af} nf={n_nf}')
    pd.DataFrame(stride_rows).to_csv(DATA_DIR / 'stride_counts.csv', index=False)
    return pd.DataFrame(rows)


def add_ailr(df):
    df = df.copy()
    spatio = ['strideLength', 'strideTime', 'stanceTime', 'swingTime', 'stanceDuration']
    joints = [f'{j}_{m}' for j in ['Ankle', 'Knee', 'Hip']
              for m in ['maxData_st', 'maxData_sw', 'minData_st', 'minData_sw']]
    aff_cols  = [f'aff_{p}'  for p in spatio] + [f'aff_{j}'  for j in joints]
    npar_cols = [f'npar_{p}' for p in spatio] + [f'npar_{j}' for j in joints]
    lr_rows = []
    for _, row in df.iterrows():
        d = {'patient_id': row['patient_id'], 'paretic_side': row['paretic_side'],
             'rdv': row['rdv'], 'walkingSpeed': row['walkingSpeed']}
        is_left = row['paretic_side'] == 'left'
        for ac, nc in zip(aff_cols, npar_cols):
            base = ac.replace('aff_', '')
            d[f'left_{base}']  = row[ac] if is_left else row[nc]
            d[f'right_{base}'] = row[nc] if is_left else row[ac]
        lr_rows.append(d)
    df_lr = pd.DataFrame(lr_rows)
    for ac, nc in zip(aff_cols, npar_cols):
        base = ac.replace('aff_', '')
        lv = df_lr[f'left_{base}']
        rv = df_lr[f'right_{base}']
        df_lr[f'AILR_{base}'] = (lv - rv) / (lv.abs() + rv.abs() + 1e-10)
    return df_lr


def compute_feature_stats(df_lr, feat_cols):
    rows    = []
    left_g  = df_lr[df_lr['paretic_side'] == 'left']
    right_g = df_lr[df_lr['paretic_side'] == 'right']
    for f in feat_cols:
        g0 = left_g[f].dropna().values
        g1 = right_g[f].dropna().values
        if len(g0) < 2 or len(g1) < 2:
            continue
        u, p = mannwhitneyu(g0, g1, alternative='two-sided')
        r = 1 - 2 * u / (len(g0) * len(g1))
        rows.append({'feature': f, 'p_value': p, 'effect_size_r': r,
                     'mean_left': g0.mean(), 'std_left': g0.std(),
                     'mean_right': g1.mean(), 'std_right': g1.std()})
    return pd.DataFrame(rows).sort_values('p_value').reset_index(drop=True)


def loso_classify(X, y, model, top_k):
    n     = len(y)
    preds = np.zeros(n, dtype=int)
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        X_tr, y_tr = X[tr], y[tr]
        X_te = X[i].reshape(1, -1)
        p_vals = []
        for f in range(X_tr.shape[1]):
            g0 = X_tr[y_tr == 0, f]
            g1 = X_tr[y_tr == 1, f]
            if len(g0) < 2 or len(g1) < 2:
                p_vals.append(1.0)
            else:
                _, p = mannwhitneyu(g0, g1, alternative='two-sided')
                p_vals.append(p)
        sel    = np.argsort(p_vals)[:top_k]
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr[:, sel])
        X_te_s = scaler.transform(X_te[:, sel])
        try:
            m = type(model)(**model.get_params())
            m.fit(X_tr_s, y_tr)
            preds[i] = m.predict(X_te_s)[0]
        except Exception:
            preds[i] = int(np.bincount(y_tr).argmax())
    return preds


def split_and_save(fig, stem):
    full = FIG_DIR / f'{stem}.png'
    fig.savefig(str(full), dpi=150, bbox_inches='tight')
    plt.close(fig)
    img  = Image.open(full)
    w, h = img.size
    mid  = w // 2
    img.crop((0, 0, mid, h)).save(FIG_DIR / f'{stem}_a.png', dpi=(150, 150))
    img.crop((mid, 0, w, h)).save(FIG_DIR / f'{stem}_b.png', dpi=(150, 150))


def compute_psd(x, fs=100):
    N      = len(x)
    window = np.hanning(N)
    fft_v  = np.fft.rfft(x * window)
    power  = (np.abs(fft_v) ** 2) / (np.sum(window ** 2) * fs)
    power[1:-1] *= 2
    return np.fft.rfftfreq(N, d=1.0 / fs), power


def swing_pulse(t, t0, amp, dip_amp=-320):
    sig  = np.zeros_like(t)
    rise = 0.15; fall = 0.20; neg = 0.12
    mask_pos = (t >= t0) & (t < t0 + rise + fall)
    sig[mask_pos] = amp * np.sin(np.pi * (t[mask_pos] - t0) / (rise + fall)) ** 2
    t_neg    = t0 + rise + fall
    mask_neg = (t >= t_neg) & (t < t_neg + neg)
    sig[mask_neg] = dip_amp * np.sin(np.pi * (t[mask_neg] - t_neg) / neg) ** 1.5
    return sig


def fig_dataset_overview(df_lr):
    counts    = df_lr['paretic_side'].value_counts()
    rdv_dist  = df_lr['rdv'].value_counts().sort_index()
    rdv_labels = [r.replace('RDV', 'Session ') for r in rdv_dist.index]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    bars0 = axes[0].bar(counts.index, counts.values,
                        color=['#e74c3c' if x == 'left' else '#3498db' for x in counts.index],
                        edgecolor='black', linewidth=0.8)
    for bar, v in zip(bars0, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, v + 0.15,
                     str(v), ha='center', fontweight='bold', fontsize=13)
    axes[0].set_ylabel('Number of Patients', fontsize=11)
    axes[0].set_ylim(0, max(counts.values) + 3)
    axes[0].grid(axis='y', alpha=0.3)
    bars1 = axes[1].bar(range(len(rdv_dist)), rdv_dist.values,
                        color='#95a5a6', edgecolor='black', linewidth=0.8)
    for bar, v in zip(bars1, rdv_dist.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, v + 0.1,
                     str(v), ha='center', fontweight='bold', fontsize=12)
    axes[1].set_xticks(range(len(rdv_dist)))
    axes[1].set_xticklabels(rdv_labels, fontsize=10)
    axes[1].set_ylabel('Number of Patients', fontsize=11)
    axes[1].set_ylim(0, max(rdv_dist.values) + 3)
    axes[1].grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(str(FIG_DIR / 'dataset_overview.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  dataset_overview.png')


def fig_feature_statistics(df_stats, top_n=20):
    top = df_stats.head(top_n).copy()
    top['label'] = top['feature'].str.replace('_', ' ')
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    colors_p = ['#e74c3c' if p < 0.05 else '#95a5a6' for p in top['p_value']]
    axes[0].barh(range(len(top)), -np.log10(top['p_value'] + 1e-10),
                 color=colors_p, edgecolor='white', linewidth=0.5)
    axes[0].axvline(-np.log10(0.05), color='black', linestyle='--', linewidth=1.2, label='p = 0.05')
    axes[0].set_yticks(range(len(top)))
    axes[0].set_yticklabels(top['label'], fontsize=8)
    axes[0].set_xlabel('-log₁₀(p-value)', fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', alpha=0.3)
    colors_e = ['#e74c3c' if r > 0 else '#3498db' for r in top['effect_size_r']]
    axes[1].barh(range(len(top)), top['effect_size_r'],
                 color=colors_e, edgecolor='white', linewidth=0.5)
    axes[1].axvline(0, color='black', linewidth=0.8)
    axes[1].set_yticks(range(len(top)))
    axes[1].set_yticklabels(top['label'], fontsize=8)
    axes[1].set_xlabel('Rank-biserial effect size r', fontsize=10)
    axes[1].invert_yaxis()
    axes[1].grid(axis='x', alpha=0.3)
    plt.tight_layout()
    split_and_save(fig, 'feature_statistics')
    print('  feature_statistics.png + _a / _b')


def fig_top_feature_distributions(df_lr, df_stats):
    top_feats = df_stats.head(2)['feature'].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    colors = {'left': '#e74c3c', 'right': '#3498db'}
    for ax, feat in zip(axes, top_feats):
        sides = ['left', 'right']
        data  = [df_lr[df_lr['paretic_side'] == s][feat].values for s in sides]
        parts = ax.violinplot(data, positions=[0, 1], showmedians=True, showextrema=True)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[sides[i]]); pc.set_alpha(0.5)
        parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(2)
        for i, (s, d) in enumerate(zip(sides, data)):
            jitter = np.random.default_rng(i).uniform(-0.08, 0.08, len(d))
            ax.scatter(np.full(len(d), i) + jitter, d,
                       color=colors[s], s=45, alpha=0.85, zorder=3,
                       edgecolors='white', linewidth=0.4)
        row_s = df_stats[df_stats['feature'] == feat].iloc[0]
        stars = ('***' if row_s['p_value'] < 0.001 else
                 ('**' if row_s['p_value'] < 0.01 else
                  ('*'  if row_s['p_value'] < 0.05 else 'ns')))
        ax.set_title(f'{feat.replace("_", " ")}\n'
                     f'p={row_s["p_value"]:.4f} {stars}  r={row_s["effect_size_r"]:.2f}',
                     fontsize=8, fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Left paretic', 'Right paretic'], fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(str(FIG_DIR / 'top_feature_distributions.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  top_feature_distributions.png')


def fig_confusion_matrices(results_k2, le):
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    for ax, (name, res) in zip(axes.flatten('F'), results_k2.items()):
        ConfusionMatrixDisplay(res['cm'], display_labels=le.classes_).plot(
            ax=ax, colorbar=False, cmap='Blues')
        ax.set_title(f'{name}\nAcc={res["acc"]:.2f}  Bal={res["bal_acc"]:.2f}',
                     fontsize=10, fontweight='bold')
    plt.tight_layout()
    split_and_save(fig, 'confusion_matrices')
    print('  confusion_matrices.png + _a / _b')


def fig_model_comparison(results_k2):
    names      = list(results_k2.keys())
    bal_accs   = [results_k2[n]['bal_acc'] for n in names]
    fig, ax    = plt.subplots(figsize=(7, 4))
    bar_colors = ['#1A5276', '#2E86AB', '#717D7E', '#B2BABB']
    bars = ax.bar(names, bal_accs, color=bar_colors, width=0.55,
                  edgecolor='white', linewidth=1.2)
    for bar, v in zip(bars, bal_accs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f'{v:.2f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.6, label='Chance (0.5)')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Balanced Accuracy (LOSO-CV)', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(str(FIG_DIR / 'model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  model_comparison.png')


def fig_ksweep(sweep_results, k_values):
    fig, ax   = plt.subplots(figsize=(8, 4.5))
    colors_sw = {'SVM Linear': '#E74C3C', 'Logistic Reg': '#2980B9'}
    for name, vals in sweep_results.items():
        ax.plot(k_values, vals, marker='o', markersize=6, linewidth=2.2,
                color=colors_sw[name], label=name)
    ax.axvline(2, color='#27AE60', linewidth=1.8, linestyle='--', alpha=0.8, label='k = 2 (chosen)')
    ax.scatter([2, 2],
               [sweep_results['SVM Linear'][1], sweep_results['Logistic Reg'][1]],
               s=90, zorder=5, color='#27AE60')
    ax.set_xlabel('Number of features selected per fold (k)', fontsize=12)
    ax.set_ylabel('Balanced accuracy (LOSO-CV)', fontsize=12)
    ax.set_ylim(0.45, 1.02)
    ax.set_xticks(k_values)
    ax.axhline(0.5, color='grey', linewidth=0.9, linestyle=':', alpha=0.5, label='Chance (0.5)')
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fig.savefig(str(FIG_DIR / 'feature_selection_frequency.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  feature_selection_frequency.png')


def fig_permutation(obs_k18, null_k18, p_k18, obs_k2, null_k2, p_k2, n_feats):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, obs, null, name, p, col in zip(
            axes,
            [obs_k18, obs_k2],
            [null_k18, null_k2],
            [f'SVM Linear — full features (k={n_feats})', 'SVM Linear — top-2 features (k=2)'],
            [p_k18, p_k2],
            ['#e74c3c', '#3498db']):
        ax.hist(null, bins=25, color='#bdc3c7', edgecolor='white',
                label=f'Null distribution (n={len(null)})')
        ax.axvline(obs, color=col, linewidth=2.5, label=f'Observed = {obs:.3f}')
        ax.axvline(0.5, color='grey', linewidth=1.2, linestyle='--', label='Chance (0.5)')
        sig = ('***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns')))
        ax.set_title(f'{name}\np = {p:.4f}  {sig}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Balanced Accuracy (permuted labels)', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    split_and_save(fig, 'permutation_test')
    print('  permutation_test.png + _a / _b')


def fig_psd():
    FS = 100; FC = 10
    df_raw   = pd.read_csv(RAW_PSD_CSV)
    shank_df = df_raw[df_raw['sensor'] == 2].reset_index(drop=True)
    gyr_norm = np.sqrt(shank_df['gyr_x']**2 + shank_df['gyr_y']**2
                       + shank_df['gyr_z']**2).values
    f, p      = compute_psd(gyr_norm, FS)
    df_bin    = f[1] - f[0]
    cum       = np.cumsum(p) * df_bin
    frac      = cum / cum[-1] * 100
    fc_idx    = np.searchsorted(f, FC)
    pct_below = frac[fc_idx]
    f_lim    = f[f <= 20]; p_lim = p[f <= 20]; frac_lim = frac[f <= 20]
    mask_lo  = f_lim <= FC; mask_hi = f_lim >= FC
    fig, ax  = plt.subplots(figsize=(7, 4.5))
    ax.fill_between(f_lim[mask_lo], p_lim[mask_lo], alpha=0.50, color='#2ECC71',
                    zorder=2, label='Gait content  (0–10 Hz)')
    ax.fill_between(f_lim[mask_hi], p_lim[mask_hi], alpha=0.25, color='#95A5A6',
                    zorder=2, label='Noise  (>10 Hz)')
    ax.plot(f_lim, p_lim, color='#2980B9', linewidth=2.2, zorder=3, label='Gyroscope PSD')
    ax.axvline(FC, color='#C0392B', linewidth=2.0, linestyle='--', zorder=4,
               label=f'Filter cutoff  ({FC} Hz)')
    ax2 = ax.twinx()
    ax2.plot(f_lim, frac_lim, color='#8E44AD', linewidth=2.0, linestyle=':', alpha=0.90,
             zorder=5, label='Cumulative energy (%)')
    ax2.axhline(pct_below, color='#8E44AD', linewidth=0.8, linestyle=':', alpha=0.40)
    ax2.set_ylim(0, 115); ax2.set_yticks([0, 25, 50, 75, 100])
    ax2.set_ylabel('Cumulative energy (%)', color='#8E44AD', fontsize=11)
    ax2.tick_params(axis='y', labelcolor='#8E44AD', labelsize=10)
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_color('#8E44AD')
    ax.text(1.15, 0.97, '1 Hz\n(step rate)', transform=ax.get_xaxis_transform(),
            fontsize=8.5, color='#E67E22', va='top', ha='left', style='italic')
    ax.set_xlim(0, 20)
    ax.set_xlabel('Frequency (Hz)', fontsize=13)
    ax.set_ylabel('Power Spectral Density  (rad/s)² / Hz', fontsize=11)
    ax.tick_params(labelsize=11)
    ax.spines['top'].set_visible(False)
    ax.grid(True, which='major', alpha=0.20, axis='x')
    ax.text(0.25, 0.06, 'Gait harmonics\n(1 – 9 Hz)', transform=ax.transAxes,
            ha='center', va='bottom', fontsize=9, color='#1E8449', style='italic')
    ax.text(0.78, 0.06, 'Noise floor', transform=ax.transAxes,
            ha='center', va='bottom', fontsize=9, color='#7F8C8D', style='italic')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9.5, framealpha=0.92,
              edgecolor='#CCCCCC', loc='center right', handlelength=1.8)
    plt.tight_layout()
    fig.savefig(str(FIG_DIR / 'psd_filter_justification.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)
    print('  psd_filter_justification.png')


def fig_swing_thresholds():
    np.random.seed(42)
    t  = np.linspace(0, 10, 1000)
    Dh = 100; Dl = 30; Ds = -300
    swing_starts = [0.6, 1.7, 3.3, 4.4]
    swing_amps   = [240, 210, 250, 220]
    D = np.zeros_like(t)
    for t0, amp in zip(swing_starts, swing_amps):
        D += swing_pulse(t, t0, amp)
    D += 15 * np.sin(2 * np.pi * 0.4 * t) + np.random.normal(0, 8, len(t))
    D  = np.clip(D, -380, 310)
    swings           = [(0.60, 0.96), (1.70, 2.06), (3.30, 3.66), (4.40, 4.76)]
    swing_labels_txt = ['Swing 1', 'Swing 2', 'Swing 3', 'Swing 4']
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor('#F7F9FC'); ax.set_facecolor('#F7F9FC')
    for (ts, te) in swings:
        ax.axvspan(ts, te, alpha=0.6, color='#D5EEF5', zorder=1)
    ax.plot(t, D, color='#1B3A6B', linewidth=1.8, zorder=3, label='D signal')
    ax.axhline(Dh, color='#E74C3C', linewidth=2.0, linestyle='--', zorder=4)
    ax.axhline(Dl, color='#F39C12', linewidth=2.0, linestyle='--', zorder=4)
    ax.axhline(Ds, color='#8E44AD', linewidth=2.0, linestyle='--', zorder=4)
    ax.axhline(0,  color='#888888', linewidth=0.8, linestyle='-',  zorder=2)
    x_label = 9.85
    ax.text(x_label, Dh + 8,  'Dh = 100',  color='#E74C3C', fontsize=13, fontweight='bold', va='bottom')
    ax.text(x_label, Dl + 8,  'Dl = 30',   color='#F39C12', fontsize=13, fontweight='bold', va='bottom')
    ax.text(x_label, Ds - 8,  'Ds = −300', color='#8E44AD', fontsize=13, fontweight='bold', va='top')
    label_y = 302
    for (ts, te), lbl in zip(swings, swing_labels_txt):
        ax.text((ts + te) / 2, label_y, lbl, ha='center', va='top',
                fontsize=10, color='#1B6B8A', fontweight='bold', linespacing=1.3)
    ts1, te1 = swings[0]; brace_y = 277
    ax.annotate('', xy=(te1, brace_y), xytext=(ts1, brace_y),
                arrowprops=dict(arrowstyle='<->', color='#C0392B', lw=2.0))
    ax.text((ts1 + te1) / 2, brace_y - 14, 'Tm (min duration)',
            ha='center', fontsize=11, color='#C0392B', fontweight='bold', va='top')
    _, te2 = swings[1]; ts3, _ = swings[2]; td_y = -240
    ax.annotate('', xy=(ts3, td_y), xytext=(te2, td_y),
                arrowprops=dict(arrowstyle='<->', color='#8E44AD', lw=2.0))
    ax.text((te2 + ts3) / 2, td_y - 20, 'Td (min gap)',
            ha='center', fontsize=11, color='#8E44AD', fontweight='bold', va='top')
    ax.plot([te2, te2], [td_y, Ds - 5], color='#8E44AD', lw=1.2, linestyle=':', zorder=3)
    ax.plot([ts3, ts3], [td_y, Ds - 5], color='#8E44AD', lw=1.2, linestyle=':', zorder=3)
    idx_dh = np.where((t > swings[2][0]) & (D > Dh))[0]
    if len(idx_dh):
        ax.scatter(t[idx_dh[0]], Dh, s=60, zorder=6, color='#E74C3C')
        ax.annotate('D > Dh\n(swing starts)', xy=(t[idx_dh[0]], Dh),
                    xytext=(t[idx_dh[0]] - 0.65, Dh + 80), fontsize=10, color='#E74C3C',
                    arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=1.5))
    idx_dl = np.where((t > swings[2][0] + 0.15) & (D < Dl) & (t < swings[2][1] + 0.2))[0]
    if len(idx_dl):
        ax.scatter(t[idx_dl[0]], Dl, s=60, zorder=6, color='#F39C12')
        ax.annotate('D < Dl\n(swing ends)', xy=(t[idx_dl[0]], Dl),
                    xytext=(t[idx_dl[0]] + 0.25, Dl + 90), fontsize=10, color='#F39C12',
                    arrowprops=dict(arrowstyle='->', color='#F39C12', lw=1.5))
    idx_ds = np.where((t > swings[0][1] - 0.05) & (D < Ds + 30))[0]
    if len(idx_ds):
        ax.scatter(t[idx_ds[0]], D[idx_ds[0]], s=60, zorder=6, color='#8E44AD')
        ax.annotate('D < Ds\n(required negative dip)', xy=(t[idx_ds[0]], D[idx_ds[0]]),
                    xytext=(t[idx_ds[0]] + 1.2, -180), fontsize=10, color='#8E44AD',
                    arrowprops=dict(arrowstyle='->', color='#8E44AD', lw=1.5))
    legend_elements = [
        mpatches.Patch(color='#D5EEF5', label='Detected swing phase', alpha=0.8),
        plt.Line2D([0], [0], color='#E74C3C', lw=2, linestyle='--', label='Dh — high threshold'),
        plt.Line2D([0], [0], color='#F39C12', lw=2, linestyle='--', label='Dl — low threshold'),
        plt.Line2D([0], [0], color='#8E44AD', lw=2, linestyle='--', label='Ds — negative threshold'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
              framealpha=0.85, edgecolor='#CCCCCC')
    ax.set_xlim(0, 10.3); ax.set_ylim(-400, 330)
    ax.set_xlabel('Time (s)', fontsize=13)
    ax.set_ylabel('Foot acceleration norm  (a.u.)', fontsize=13)
    ax.tick_params(labelsize=11)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fig.savefig(str(FIG_DIR / 'swing_detection_thresholds.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)
    print('  swing_detection_thresholds.png')


# ── main ──────────────────────────────────────────────────────────────────────

config    = load_config()
labels    = pd.read_csv(LABELS_CSV)
label_map = dict(zip(labels['patient_id'], labels['paretic_side']))

print(f'[1/6] Extracting features for {len(config)} patients...')
df_raw = build_feature_matrix(config, label_map)
df_raw.to_csv(DATA_DIR / 'features.csv', index=False)

print('[2/6] Computing AILR features...')
df_lr     = add_ailr(df_raw)
df_lr.to_csv(DATA_DIR / 'features_ailr.csv', index=False)
AILR_COLS = [c for c in df_lr.columns if c.startswith('AILR_')]
FEAT_COLS = ['walkingSpeed'] + AILR_COLS
X         = df_lr[FEAT_COLS].values.astype(float)
le        = LabelEncoder()
y         = le.fit_transform(df_lr['paretic_side'].values)

print('[3/6] Feature statistics (Mann-Whitney)...')
df_stats = compute_feature_stats(df_lr, FEAT_COLS)
df_stats.to_csv(DATA_DIR / 'feature_statistics.csv', index=False)

print('[4/6] LOSO classification (k=2, 4 models)...')
results_k2   = {}
pred_records = {'patient_id': df_lr['patient_id'].values,
                'true_label': le.inverse_transform(y)}
for name, model in MODELS.items():
    preds   = loso_classify(X, y, model, top_k=2)
    acc     = (preds == y).mean()
    bal_acc = balanced_accuracy_score(y, preds)
    cm      = confusion_matrix(y, preds)
    results_k2[name] = {'preds': preds, 'cm': cm, 'acc': acc, 'bal_acc': bal_acc}
    pred_records[name] = le.inverse_transform(preds)
    print(f'  {name:<16} acc={acc:.3f}  bal_acc={bal_acc:.3f}')

pd.DataFrame(pred_records).to_csv(RESULTS_DIR / 'LOSO_predictions.csv', index=False)

n_feats   = len(FEAT_COLS)
obs_k18   = balanced_accuracy_score(y, loso_classify(X, y, MODELS['SVM Linear'], top_k=n_feats))
comp_rows = [{'model': n, 'acc': round(r['acc'], 4), 'bal_acc': round(r['bal_acc'], 4)}
             for n, r in results_k2.items()]
comp_rows.append({'model': f'SVM Linear (k={n_feats})', 'acc': round(obs_k18, 4), 'bal_acc': round(obs_k18, 4)})
pd.DataFrame(comp_rows).to_csv(RESULTS_DIR / 'model_comparison.csv', index=False)

print('[5/6] k-sweep + permutation tests...')
k_values     = list(range(1, n_feats + 1))
sweep_models = {'SVM Linear': MODELS['SVM Linear'], 'Logistic Reg': MODELS['Logistic Reg']}
sweep_results = {n: [] for n in sweep_models}
for k in k_values:
    for name, model in sweep_models.items():
        preds = loso_classify(X, y, model, top_k=k)
        sweep_results[name].append(balanced_accuracy_score(y, preds))
    print(f'  k={k:2d}  SVM={sweep_results["SVM Linear"][-1]:.3f}  LR={sweep_results["Logistic Reg"][-1]:.3f}')
pd.DataFrame({'k': k_values, **sweep_results}).to_csv(RESULTS_DIR / 'ksweep_results.csv', index=False)

svm      = MODELS['SVM Linear']
null_k18 = []
null_k2  = []
n_perm   = 200
rng      = np.random.default_rng(42)
print(f'  Permutation k={n_feats} (full features)...')
for i in range(n_perm):
    y_p = rng.permutation(y)
    null_k18.append(balanced_accuracy_score(y_p, loso_classify(X, y_p, svm, top_k=n_feats)))
    if (i + 1) % 50 == 0:
        print(f'    {i+1}/{n_perm} done')
null_k18 = np.array(null_k18)

rng2 = np.random.default_rng(42)
print('  Permutation k=2...')
for i in range(n_perm):
    y_p = rng2.permutation(y)
    null_k2.append(balanced_accuracy_score(y_p, loso_classify(X, y_p, svm, top_k=2)))
    if (i + 1) % 50 == 0:
        print(f'    {i+1}/{n_perm} done')
null_k2 = np.array(null_k2)
pd.DataFrame({'null_k18': null_k18, 'null_k2': null_k2}).to_csv(
    RESULTS_DIR / 'permutation_null.csv', index=False)

obs_k2 = results_k2['SVM Linear']['bal_acc']
p_k18  = (null_k18 >= obs_k18).mean()
p_k2   = (null_k2  >= obs_k2).mean()
print(f'  SVM Linear k={n_feats}: obs={obs_k18:.3f}  p={p_k18:.4f}')
print(f'  SVM Linear k=2:         obs={obs_k2:.3f}  p={p_k2:.4f}')

print('[6/6] Generating figures...')
fig_dataset_overview(df_lr)
fig_feature_statistics(df_stats)
fig_top_feature_distributions(df_lr, df_stats)
fig_confusion_matrices(results_k2, le)
fig_model_comparison(results_k2)
fig_ksweep(sweep_results, k_values)
fig_permutation(obs_k18, null_k18, p_k18, obs_k2, null_k2, p_k2, n_feats)
fig_psd()
fig_swing_thresholds()

print(f'\nDone. Results -> {RESULTS_DIR}\nFigures  -> {FIG_DIR}')
