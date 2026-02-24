"""
Main Gait Processing Functions
"""

import numpy as np
from .gait_functions import *


def extraction_gait_parm_modified(step_time_data, data, time_vec, swing):
    """
    Calculate gait parameters for both affected and non-affected legs

    Parameters:
    -----------
    step_time_data : dict
        Contains 'af' and 'nf' step timing arrays
    data : dict
        Sensor data structure
    time_vec : array
        Time vector in milliseconds
    swing : dict
        Swing phase signals for 'af' and 'nf'

    Returns:
    --------
    out : dict
        Gait parameters including walking speed and leg-specific metrics
    """
    # Calculate time intervals (dt = 10 ms)
    dT = np.ones(len(time_vec)) * 0.01

    # Compute overall walking speed from waist sensor
    waist_sensor = data['cal'][6]  # Index 6 for waist (0-indexed)
    pos_wa = np.array([0.0, 0.0])
    vel_wa = np.array([0.0, 0.0])

    for j in range(len(time_vec)):
        dt = dT[j]
        rot = quat2rot(waist_sensor['quat'][j, :])
        acc_corr = (rot @ waist_sensor['acc'][j, :] - np.array([0, 0, 9.8]))
        vel_wa = vel_wa + acc_corr[0:2] * dt
        pos_wa = pos_wa + vel_wa * dt

    totalTime = (time_vec[-1] - time_vec[0]) / 1000
    walkingSpeed_overall = np.linalg.norm(pos_wa) / totalTime

    # Offset correction for step length
    offset = 0.25

    # Compute parameters for affected and non-affected legs
    af_params = compute_gait_params(
        data['cal'][0], step_time_data['af'], time_vec, swing['af'], dT, offset
    )
    nf_params = compute_gait_params(
        data['cal'][3], step_time_data['nf'], time_vec, swing['nf'], dT, offset
    )

    out = {
        'walkingSpeed': walkingSpeed_overall,
        'af': af_params,
        'nf': nf_params
    }

    return out


def compute_gait_params(sensor, step_data, time_vec, swing_signal, dT, offset):
    """
    Calculate gait parameters from sensor data over each step

    Parameters:
    -----------
    sensor : dict
        Sensor data with quat, acc fields
    step_data : array
        Nx3 matrix [stanceStart, stanceEnd, swingEnd]
    time_vec : array
        Time vector in milliseconds
    swing_signal : array
        Binary swing phase signal
    dT : array
        Time intervals in seconds
    offset : float
        Offset for step length (meters)

    Returns:
    --------
    gait_params : dict
        Stride length, time, stance/swing durations
    """
    N = step_data.shape[0]

    step_length = np.zeros(N)
    swing_time = np.zeros(N)
    stance_time = np.zeros(N)
    stride_time = np.zeros(N)
    stance_duration = np.zeros(N)

    for i in range(N):
        idx_start = int(step_data[i, 0])
        idx_stance_end = int(step_data[i, 1])
        idx_swing_end = int(step_data[i, 2])

        # Calculate time durations
        strideT = (time_vec[idx_swing_end] - time_vec[idx_start]) / 1000
        stanceT = (time_vec[idx_stance_end] - time_vec[idx_start]) / 1000
        swingT = (time_vec[idx_swing_end] - time_vec[idx_stance_end]) / 1000

        # Initialize position and velocity
        pos = np.array([0.0, 0.0])
        vel = np.array([0.0, 0.0])

        for j in range(idx_start, idx_swing_end + 1):
            dt = dT[j]
            rot = quat2rot(sensor['quat'][j, :])
            acc_corr = (rot @ sensor['acc'][j, :] - np.array([0, 0, 9.8]))

            # Update velocity
            vel = vel + acc_corr[0:2] * dt

            # Reset velocity during stance
            if swing_signal[j] == 0:
                vel = np.array([0.0, 0.0])

            # Update position
            pos = pos + vel * dt

        # Final position update
        pos = pos + vel * dt

        # Store parameters
        step_length[i] = np.linalg.norm(pos) + offset
        stride_time[i] = strideT
        stance_time[i] = stanceT
        swing_time[i] = swingT
        stance_duration[i] = (stanceT / strideT) * 100

    gait_params = {
        'strideLength': step_length,
        'strideTime': stride_time,
        'stanceTime': stance_time,
        'swingTime': swing_time,
        'stanceDuration': stance_duration
    }

    return gait_params


def process_gait_data(calib_data, calib_indices, walk_data, walk_indices, swing_params=None, return_debug=False):
    """
    Main processing function for gait data

    Parameters:
    -----------
    calib_data : dict
        Calibration data structure
    calib_indices : dict
        Calibration index ranges
    walk_data : dict
        Walking data structure
    walk_indices : dict
        Walking index ranges
    swing_params : list, optional
        Swing detection parameters [Dh, Dl, Ds, Tm, Td]
        Default: [100, 30, -300, 30, 10]

    Returns:
    --------
    output : dict
        Processed gait parameters, characteristics, and cyclic data
    """
    fs = 100
    fc_qua = 10
    fc_imu = 10

    # 1. Data Preparation
    # Handle both array and scalar index types
    calib_start = calib_indices['indexStart'][0] if isinstance(calib_indices['indexStart'], np.ndarray) else calib_indices['indexStart']
    calib_end = calib_indices['indexEnd'][0] if isinstance(calib_indices['indexEnd'], np.ndarray) else calib_indices['indexEnd']
    calib_idx = np.array([[calib_start, calib_end]])

    walk_start = walk_indices['indexStart'][0] if isinstance(walk_indices['indexStart'], np.ndarray) else walk_indices['indexStart']
    walk_end = walk_indices['indexEnd'][0] if isinstance(walk_indices['indexEnd'], np.ndarray) else walk_indices['indexEnd']
    walk_idx = np.array([[walk_start, walk_end]])

    walk_range = range(walk_idx[0, 0], walk_idx[0, 1])

    # Calibration Data (quaternions)
    quat = [calib_data['cal'][i]['quat'] for i in range(7)]
    quat = filter_quat_lowpass(quat, fs, fc_qua, 6, True)

    # Walking Data - keep UNFILTERED quaternions for gait parameters (like MATLAB does)
    walk_data_unfiltered = {'cal': []}
    for i in range(7):
        sensor_data = {
            'quat': walk_data['cal'][i]['quat'][walk_range, :],
            'acc': walk_data['cal'][i]['acc'][walk_range, :],
            'gyr': walk_data['cal'][i]['gyr'][walk_range, :]
        }
        walk_data_unfiltered['cal'].append(sensor_data)

    # Extract for filtering
    walk_quat = [walk_data_unfiltered['cal'][i]['quat'] for i in range(7)]
    walk_acc = [walk_data_unfiltered['cal'][i]['acc'] for i in range(7)]
    walk_gyr = [walk_data_unfiltered['cal'][i]['gyr'] for i in range(7)]

    # Filter quaternions and gyro (for joint angles only)
    walk_quat_filt = filter_quat_lowpass(walk_quat, fs, fc_qua, 6, True)
    walk_gyr_filt = filter_quat_lowpass(walk_gyr, fs, fc_imu, 4, False)

    # Create filtered data structure (for swing detection and joint angles)
    walk_data_filtered = {'cal': []}
    for i in range(7):
        sensor_data = {
            'quat': walk_quat_filt[i],
            'acc': walk_acc[i],  # Acceleration not filtered
            'gyr': walk_gyr_filt[i]
        }
        walk_data_filtered['cal'].append(sensor_data)

    # 2. Calibration (Hybrid: Static Gravity + Walking PCA)
    R_SB = []
    for i in range(7):
        R_SB.append(calibration_z(quat[i], calib_idx, walk_gyr_filt[i]))

    # 3. Joint Angle Calculation (uses FILTERED quaternions)
    seq = 'ZXZ'
    r, p, y = [], [], []

    # Non-affected leg joints
    r_tmp, p_tmp, y_tmp = calculate_joint_angles(walk_quat_filt[1], walk_quat_filt[0], R_SB[1], R_SB[0], seq)
    r.append(r_tmp); p.append(p_tmp); y.append(y_tmp)

    r_tmp, p_tmp, y_tmp = calculate_joint_angles(walk_quat_filt[2], walk_quat_filt[1], R_SB[2], R_SB[1], seq)
    r.append(r_tmp); p.append(p_tmp); y.append(y_tmp)

    r_tmp, p_tmp, y_tmp = calculate_joint_angles(walk_quat_filt[6], walk_quat_filt[2], R_SB[6], R_SB[2], seq)
    r.append(r_tmp); p.append(p_tmp); y.append(y_tmp)

    # Affected leg joints
    r_tmp, p_tmp, y_tmp = calculate_joint_angles(walk_quat_filt[4], walk_quat_filt[3], R_SB[4], R_SB[3], seq)
    r.append(r_tmp); p.append(p_tmp); y.append(y_tmp)

    r_tmp, p_tmp, y_tmp = calculate_joint_angles(walk_quat_filt[5], walk_quat_filt[4], R_SB[5], R_SB[4], seq)
    r.append(r_tmp); p.append(p_tmp); y.append(y_tmp)

    r_tmp, p_tmp, y_tmp = calculate_joint_angles(walk_quat_filt[6], walk_quat_filt[5], R_SB[6], R_SB[5], seq)
    r.append(r_tmp); p.append(p_tmp); y.append(y_tmp)

    # Map to output structure
    tar = y
    angles = {
        'aa': -tar[0],  # Affected ankle
        'ak': tar[1],   # Affected knee
        'ah': -tar[2],  # Affected hip
        'na': -tar[3],  # Non-affected ankle
        'nk': tar[4],   # Non-affected knee
        'nh': -tar[5]   # Non-affected hip
    }

    # 4. Swing Detection & Step Timing (uses FILTERED data)
    if swing_params is None:
        set_params = [100, 30, -300, 30, 10]  # Default parameters
    else:
        set_params = swing_params
    Sw1, Sw2 = swing_detection(walk_data_filtered, set_params)

    min_length = 30
    Sw1 = remove_short_pulses(Sw1, min_length)
    Sw2 = remove_short_pulses(Sw2, min_length)

    step_time_data_af = separate_step(Sw1)
    step_time_data_nf = separate_step(Sw2)

    time_vec = np.arange(10, (len(walk_acc[6]) + 1) * 10, 10)

    step_time_data = {
        'af': step_time_data_af,
        'nf': step_time_data_nf,
        'all': np.vstack([step_time_data_af, step_time_data_nf])
    }
    step_time_data['all'] = step_time_data['all'][np.argsort(step_time_data['all'][:, 0])]

    swing = {'af': Sw1, 'nf': Sw2}

    # 5. Gait Parameters & Characteristics (uses UNFILTERED quaternions like MATLAB!)
    gait_out = extraction_gait_parm_modified(step_time_data, walk_data_unfiltered, time_vec, swing)

    gait_char = {
        'aa': extraction_gait_char(step_time_data_af, angles['aa']),
        'ak': extraction_gait_char(step_time_data_af, angles['ak']),
        'ah': extraction_gait_char(step_time_data_af, angles['ah']),
        'na': extraction_gait_char(step_time_data_nf, angles['na']),
        'nk': extraction_gait_char(step_time_data_nf, angles['nk']),
        'nh': extraction_gait_char(step_time_data_nf, angles['nh'])
    }

    filtered_gait_parm = process_struct_fields(gait_out)

    # 6. Cyclic Data Extraction
    mv = 2
    St_dur_af = filtered_gait_parm['af']['stanceDuration']['meanValue']
    St_dur_nf = filtered_gait_parm['nf']['stanceDuration']['meanValue']

    nos_st_af = int(np.round(St_dur_af) * mv)
    nos_sw_af = int((100 - np.round(St_dur_af)) * mv)
    cyclic_aa = extracting_cyclic_data(angles['aa'], step_time_data_af, nos_st_af, nos_sw_af)
    cyclic_ak = extracting_cyclic_data(angles['ak'], step_time_data_af, nos_st_af, nos_sw_af)
    cyclic_ah = extracting_cyclic_data(angles['ah'], step_time_data_af, nos_st_af, nos_sw_af)

    nos_st_nf = int(np.round(St_dur_nf) * mv)
    nos_sw_nf = int((100 - np.round(St_dur_nf)) * mv)
    cyclic_na = extracting_cyclic_data(angles['na'], step_time_data_nf, nos_st_nf, nos_sw_nf)
    cyclic_nk = extracting_cyclic_data(angles['nk'], step_time_data_nf, nos_st_nf, nos_sw_nf)
    cyclic_nh = extracting_cyclic_data(angles['nh'], step_time_data_nf, nos_st_nf, nos_sw_nf)

    Ti = np.linspace(0, 100, mv * 100)

    output = {
        'GaitParm': filtered_gait_parm,
        'GaitChar': gait_char,
        'CycleD': {
            'aa': cyclic_aa, 'ak': cyclic_ak, 'ah': cyclic_ah,
            'na': cyclic_na, 'nk': cyclic_nk, 'nh': cyclic_nh,
            'T': Ti
        }
    }

    if return_debug:
        output['_debug'] = {
            'angles': angles,           # raw joint angles time series (dict: aa,ak,ah,na,nk,nh)
            'swing': swing,             # swing detection signals (dict: af, nf - binary arrays)
            'step_time_data': step_time_data,  # stride boundaries
            'foot_acc': {               # foot accelerometer norm signals (for force plot)
                'af': walk_acc[6],      # affected foot acceleration (sensor 6)
                'nf': walk_acc[3],      # non-paretic foot acceleration (sensor 3)
            },
            'time_vec': time_vec,
        }

    return output
