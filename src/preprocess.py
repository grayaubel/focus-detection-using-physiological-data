import os
import pickle
import numpy as np
import pandas as pd
import scipy.signal as scisig
from scipy.signal import welch
from collections import Counter
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

fs_dict = {'ACC': 50, 'BVP': 64, 'EDA': 4, 'Resp': 700, 'label': 700}
label_to_task = {1: 'base', 2: 'tsst', 3: 'fun', 4: 'meditation'}
subject_id = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
WINDOW_IN_SECONDS = [30, 60, 300, 600]

main_path = "../data/processed/WESAD/feature_extracted/"

class SubjectData:
    def __init__(self, main_path, subject_number):
        self.name = f'S{subject_number}'
        with open(os.path.join(main_path, self.name, self.name + '.pkl'), 'rb') as file:
            self.data = pickle.load(file, encoding='latin1')
        self.bvp = self.data['signal']['wrist']['BVP']
        self.acc = self.data['signal']['wrist']['ACC']
        self.eda = self.data['signal']['wrist']['EDA']
        self.resp = self.data['signal']['chest']['Resp']
        self.labels = self.data['label']

def quest_load(subject_id):
    subject_path = f'../data/raw/WESAD/S{subject_id}/S{subject_id}_quest.csv'

    with open(subject_path, 'r') as f:
        lines = f.readlines()

    order_line = [l for l in lines if l.startswith('# ORDER')][0]
    order = order_line.strip().split(';')[1:-1]
    order = [o.strip().lower() for o in order if o.strip() not in ['bread','fread','sread']]

    stai_dict = {}
    valence_dict = {}
    arousal_dict = {}

    stai_lines = [l for l in lines if l.startswith('# STAI')]
    dim_lines = [l for l in lines if l.startswith('# DIM')]

    stai_score = np.array([[int(i) for i in l.strip().split(';')[1:-1] if i.strip().isdigit()] for l in stai_lines])
    dim_score = np.array([[int(i) for i in l.strip().split(';')[1:-1] if i.strip().isdigit()] for l in dim_lines])

    # DIM score
    valence_score = dim_score[:, 0]
    arousal_score = dim_score[:, 1]

    min_len = min(len(order), len(valence_score))

    valence_score = valence_score[:min_len]
    arousal_score = arousal_score[:min_len]

    # STAI score
    for i, task in enumerate(order[:min_len]):
        if task.startswith('medi'):
            task_key = 'meditation'
        else:
            task_key = task

        positives = (stai_score[:, 0] + stai_score[:, 3] + stai_score[:, 5]) / 3
        negatives = (stai_score[:, 1] + stai_score[:, 2] + stai_score[:, 4]) / 3

        scored_stai = negatives[i] - positives[i]

        valence_dict[task_key] = valence_score[i]
        arousal_dict[task_key] = arousal_score[i]
        stai_dict[task_key] = scored_stai

    return{
        'valence': valence_dict,
        'arousal': arousal_dict,
        'stai': stai_dict
    }

def refined_label(lbl, dim_arousal, dim_valence, stai_score):
    if lbl in [1, 4]:
        if (dim_arousal <= 3) and (dim_valence >= 5):
            return 1
        else:
            return 0
        
    elif lbl == 2:
        if stai_score <= 0:
            if dim_arousal >= 5:
                return 1
            else:
                return 0
        else:
            return 0
        
    else:
        return 0

def safe_slope(x):
    if len(x) < 2 or np.all(np.isnan(x)):
        return np.nan
    try:
        return np.polyfit(range(len(x)), x, 1)[0]
    except np.linalg.LinAlgError:
        return np.nan
    
def feature_add(df):
    df['hr_diff'] = df['HR'].diff()
    df['hr_center'] = df['HR'] - df['HR'].mean()
    df['hr_slope'] = df['HR'].rolling(window=3, min_periods=2).apply(safe_slope, raw=True)
    df['hr_zscore'] = (df['HR'] - df['HR'].mean()) / df['HR'].std()

    df['sdnn_diff'] = df['SDNN'].diff()
    df['sdnn_slope'] = df['SDNN'].rolling(window=3, min_periods=2).apply(safe_slope, raw=True)
    df['sdnn_zscore'] = (df['SDNN'] - df['SDNN'].mean()) / df['SDNN'].std()

    df['acc_ratio'] = df['net_acc_std'] / df['net_acc_mean']
    df['net_acc_mean_slope'] = df['net_acc_mean'].rolling(window=3, min_periods=2).apply(safe_slope, raw=True)
    df['net_acc_mean_diff'] = df['net_acc_mean'].diff()

    df['acc_vector_magnitude_mean'] = np.sqrt(
        df['ACC_x_mean']**2 + df['ACC_y_mean']**2 + df['ACC_z_mean']**2
    )
    df['acc_vector_slope'] = df['acc_vector_magnitude_mean'].rolling(window=3, min_periods=2).apply(safe_slope, raw=True)

    df['hr_netacc_interaction'] = df['HR'] * df['net_acc_mean']
    df['hr_sdnn_ratio'] = df['HR'] / (df['SDNN'] + 1e-6)
    df['acc_hr_slope_diff'] = df['net_acc_std'] - df['hr_slope']
    df['sdnn_netacc_ratio'] = df['SDNN'] / (df['net_acc_mean'] + 1e-6)

    df['eda_hr_interaction'] = df['EDA_mean'] * df['HR']
    df['eda_resp_ratio'] = df['EDA_std'] / df['RESP_regularity']
    df['hr_resp_interaction'] = df['HR'] * df['RESP_rate']
    df['hrv_composite'] = (df['RMSSD'] + df['SDNN'] + df['pNN50']) / 3
    df['hrv_stress_index'] = df['SDNN'] / df['RMSSD']

    df['arousal_index'] = df['HR'] * df['EDA_mean'] * df['RESP_rate']

    return df

def bvp_to_hrv(bvp_signal, fs):
    # Detect peaks
    peaks, _ = scisig.find_peaks(bvp_signal, distance=int(fs * 0.4))

    if len(peaks) < 3:
        return pd.DataFrame()
    
    # คำนวณ IBI
    ibi = np.diff(peaks) / fs * 1000 # ms
    rr_diff = np.diff(ibi)
    
    # Time axis for interpolation
    ibi_time = np.cumsum(ibi) / 1000 # sec
    interp_time = np.arange(0, ibi_time[-1], 1/0.4)
    ibi_interp = np.interp(interp_time, ibi_time, ibi)

    # คำนวณ HR
    hr = (60 * 1000) / ibi # bpm

    # HRV  metrics
    rmssd = np.sqrt(np.mean(rr_diff ** 2)) if len(rr_diff) > 0 else np.nan
    sdnn = np.std(ibi) if len(ibi) > 1 else np.nan
    pNN50 = np.sum(np.abs(rr_diff) > 50) / len(rr_diff) * 100 if len(rr_diff) > 0 else np.nan

    # Frequency domain
    f, pxx = welch(ibi_interp, fs=0.4)
    lf = np.trapz(pxx[(f >= 0.04) & (f <= 0.15)], f[(f >= 0.04) & (f <= 0.15)])
    hf = np.trapz(pxx[(f > 0.15) & (f <= 0.4)], f[(f > 0.15) & (f <= 0.4)])
    lf_hf_ratio = lf / hf if hf != 0 else np.nan

    # Alighn HR/IBI timestamsp (start at 2nd Beats)
    timestamps = peaks[1:] / fs

    return pd.DataFrame({
        'timestamps': pd.to_datetime(timestamps, unit='s'),
        'HR': hr,
        'IBI': ibi,
        'RMSSD': rmssd,
        'SDNN': sdnn,
        'pNN50': pNN50,
        'lf/hf': lf_hf_ratio
    })

def extract_resp_features(resp_signal, fs):
    peaks, _ = scisig.find_peaks(resp_signal, distance=fs * 2)
    if len(peaks) < 2:
        return {'RESP_rate': np.nan, 'RESP_regularity': np.nan}
    
    ibi = np.diff(peaks) / fs
    resp_rate = 60 / np.mean(ibi) if np.mean(ibi) > 0 else np.nan
    regularity = 1 / np.std(ibi) if np.std(ibi) > 0 else np.nan
    return {
        'RESP_rate': resp_rate,
        'RESP_regularity': regularity
        }

def extract_eda_features(eda_signal):
    x = np.arange(len(eda_signal))
    slope = float(np.polyfit(x, eda_signal, 1)[0]) if len(eda_signal) > 1 else np.nan
    return {
        'EDA_mean': np.mean(eda_signal),
        'EDA_std': np.std(eda_signal),
        'EDA_slope': slope
        }

def feature_extract(subject_id, WINDOW_IN_SECONDS):
    subject = SubjectData(main_path="../data/raw/WESAD", subject_number=subject_id)

    # Signals
    bvp = subject.bvp.flatten()
    acc = subject.acc
    eda = subject.eda
    resp = subject.resp
    labels = subject.labels

    # Windows
    window_len = fs_dict['label'] * WINDOW_IN_SECONDS
    total_len = len(labels)
    n_windows = total_len // window_len

    all_window = []

    for i in range(n_windows):
        start = i * window_len
        end = (i + 1) * window_len

        # majority vote label
        label_window = labels[start:end]
        label_window = [l for l in label_window if l in [1, 2, 3, 4]]
        if len(label_window) == 0:
            continue
        label = Counter(label_window).most_common(1)[0][0]

        # --- ACC ---
        acc_window = acc[start * fs_dict['ACC'] // fs_dict['label']: end * fs_dict['ACC'] // fs_dict['label'], :]
        if acc_window.shape[0] == 0: continue
        acc_x, acc_y, acc_z = acc_window[:, 0], acc_window[:, 1], acc_window[:, 2]
        net_acc = np.sqrt(acc_x ** 2 + acc_y ** 2 + acc_z ** 2)
        acc_features = {
            'ACC_x_mean': np.mean(acc_x),
            'ACC_y_mean': np.mean(acc_y),
            'ACC_z_mean': np.mean(acc_z),
            'net_acc_mean': np.mean(net_acc),
            'net_acc_std': np.std(net_acc)
        }

        # --- BVP / HRV ---
        bvp_window = bvp[start * fs_dict['BVP'] // fs_dict['label']: end * fs_dict['BVP'] // fs_dict['label']]
        hrv_df = bvp_to_hrv(bvp_window, fs_dict['BVP'])
        if hrv_df.empty: continue
        hrv_mean = hrv_df[['HR', 'IBI', 'RMSSD', 'SDNN', 'pNN50', 'lf/hf']].mean()

        # --- EDA ---
        eda_window = eda[start * fs_dict['EDA'] // fs_dict['label']: end * fs_dict['EDA'] // fs_dict['label']]
        eda_features = extract_eda_features(eda_window)

        # --- RESP ---
        resp_window = resp[start * fs_dict['Resp'] // fs_dict['label']: end * fs_dict['Resp'] // fs_dict['label']]
        if resp_window.ndim > 1:
            resp_window = resp_window.flatten()
        resp_features = extract_resp_features(resp_window, fs_dict['Resp'])

        data = {
            **acc_features,
            **eda_features,
            **resp_features,
            'HR': hrv_mean['HR'],
            'IBI': hrv_mean['IBI'],
            'RMSSD': hrv_mean['RMSSD'],
            'SDNN': hrv_mean['SDNN'],
            'pNN50': hrv_mean['pNN50'],
            'lf/hf': hrv_mean['lf/hf'],
            'label': label,
            'subject': subject_id
        }
        all_window.append(data)

    df = pd.DataFrame(all_window)
    all_data.append(df)

if __name__ == "__main__":

    for wind_size in WINDOW_IN_SECONDS:

        print(f"Start processing window size: {wind_size}")

        all_data = []

        for id in subject_id:
            save_path = main_path + f'{wind_size}s/'
            feature_extract(id, WINDOW_IN_SECONDS=wind_size)

        data = pd.concat(all_data, axis=0)
        data_feature_addon = []

        for id in subject_id:
            df = data[data['subject'] == id].copy().reset_index(drop=True)
            df = feature_add(df)
            df.dropna(inplace=True)
            data_feature_addon.append(df)

        df_all = pd.concat(data_feature_addon, ignore_index=True)

        features = ['HR', 'EDA_mean', 'RMSSD', 'RESP_rate', 'net_acc_mean']
        mask = pd.Series(True, index=df_all.index)

        for subject in df_all['subject'].unique():
            for label in df_all['label'].unique():
                sub_df = df_all[(df_all['subject'] == subject) & (df_all['label'] == label)]
                for feature in features:
                    q1 = sub_df[feature].quantile(0.25)
                    q3 = sub_df[feature].quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    outlier_mask = (df_all['subject'] == subject) & (df_all['label'] == label) & (
                        (df_all[feature] < lower) | (df_all[feature] > upper))
                    mask[outlier_mask] = False

        df_all = df_all[mask].reset_index(drop=True)

        df_all['focus_label'] = -1

        quest_dict = {}
        for s in df_all['subject'].unique():
            quest_dict[s] = quest_load(s)

        for i, row in tqdm(df_all.iterrows(), total=len(df_all)):
            subj = int(row['subject'])
            lbl = int(row['label'])

            if lbl not in label_to_task:
                continue

            task_name = label_to_task[lbl]
            quest = quest_dict[subj]

            # ดึงคะแนนที่ต้องใช้
            dim_arousal = quest['arousal'][task_name]
            dim_valence = quest['valence'][task_name]
            stai_score = quest['stai'][task_name] if lbl == 2 else None

            # คำนวณ label
            label = refined_label(lbl, dim_arousal, dim_valence, stai_score)

            # บันทึกลง dataframe
            df_all.at[i, 'focus_label'] = label

        df_all.to_csv(f'../data/processed/WESAD/data_processed/{wind_size}s/all_data.csv')
        print(f"✅ Done generating all_data of {wind_size}s window size!")