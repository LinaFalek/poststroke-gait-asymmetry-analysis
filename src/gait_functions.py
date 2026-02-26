

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA


def quat2rot(q):
    """
    Convert quaternion to rotation matrix
    Input: q = [q0, q1, q2, q3] (w, x, y, z format)
    Output: 3x3 rotation matrix
    """
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]

    r00 = 2*(q0*q0 + q1*q1) - 1
    r01 = 2*(q1*q2 - q0*q3)
    r02 = 2*(q1*q3 + q0*q2)
    r10 = 2*(q1*q2 + q0*q3)
    r11 = 2*(q0*q0 + q2*q2) - 1
    r12 = 2*(q2*q3 - q0*q1)
    r20 = 2*(q1*q3 - q0*q2)
    r21 = 2*(q2*q3 + q0*q1)
    r22 = 2*(q0*q0 + q3*q3) - 1

    R_GS = np.array([[r00, r01, r02],
                     [r10, r11, r12],
                     [r20, r21, r22]])
    return R_GS


def meanquat(Q):
    """
    Compute mean quaternion using eigenvector method
    Input: Q = Nx4 array of quaternions [w, x, y, z]
    Output: mean quaternion (1x4)
    """
    N = Q.shape[0]
    A = np.zeros((4, 4))

    for k in range(N):
        q = Q[k, :].reshape(4, 1)
        # Ensure sign consistency
        if q[0] < 0:
            q = -q
        A += q @ q.T

    A = A / N
    eigenvalues, eigenvectors = np.linalg.eig(A)
    idx = np.argmax(eigenvalues)
    q_mean = eigenvectors[:, idx].real

    # Make first component positive
    if q_mean[0] < 0:
        q_mean = -q_mean

    return q_mean


def enforce_sign_continuity(q):
    """
    Enforce quaternion sign continuity (q and -q represent same orientation)
    """
    q_cont = q.copy()
    for n in range(1, q.shape[0]):
        if np.dot(q_cont[n-1, :], q_cont[n, :]) < 0:
            q_cont[n, :] = -q_cont[n, :]
    return q_cont


def filter_quat_lowpass(data, fs, fc, order=4, is_quat=None):
    """
    Zero-phase Butterworth low-pass filter for quaternion or numeric data

    Parameters:
    -----------
    data : array or list of arrays
        Data to filter (can be cell array)
    fs : float
        Sampling frequency
    fc : float
        Cutoff frequency
    order : int
        Filter order (must be even)
    is_quat : bool or None
        Override automatic quaternion detection

    Returns:
    --------
    data_filt : filtered data
    """
    # Handle cell arrays (list of arrays)
    if isinstance(data, list):
        data_filt = []
        for idx, d in enumerate(data):
            if is_quat is not None:
                cell_is_quat = is_quat if isinstance(is_quat, bool) else is_quat[idx]
            else:
                cell_is_quat = None
            data_filt.append(filter_quat_lowpass(d, fs, fc, order, cell_is_quat))
        return data_filt

    # Auto-detect quaternion mode
    if is_quat is None:
        is_quat = (data.shape[1] == 4 and
                   np.all(np.abs(np.linalg.norm(data, axis=1) - 1) < 0.1))

    # Enforce sign continuity for quaternions
    if is_quat:
        data = enforce_sign_continuity(data)

    # Design filter
    Wn = fc / (fs/2)
    b, a = butter(order // 2, Wn, 'low')

    data_filt = filtfilt(b, a, data, axis=0, padtype='odd', padlen=3*(max(len(b), len(a))-1))

    # Re-normalize quaternions
    if is_quat:
        norms = np.linalg.norm(data_filt, axis=1, keepdims=True)
        data_filt = data_filt / norms

    return data_filt


def calibration_z(static_quat_data, calib_intervals, walk_gyro):
    """
    Hybrid calibration: Static gravity + Walking PCA

    Parameters:
    -----------
    static_quat_data : array
        Quaternion data from static calibration
    calib_intervals : array
        [start, end] indices for calibration
    walk_gyro : array
        Gyroscope data from walking

    Returns:
    --------
    calib_quat : Rotation object
        Calibration quaternion (sensor to body frame)
    """
    # Gravity axis from static calibration
    q_static_segment = static_quat_data[calib_intervals[0, 0]:calib_intervals[0, 1], :]
    qg_S = meanquat(q_static_segment)
    R_sw = Rotation.from_quat([qg_S[1], qg_S[2], qg_S[3], qg_S[0]]).as_matrix()

    # Sensor frame gravity direction
    Y_A = np.linalg.inv(R_sw) @ np.array([0, 0, 1])
    Y_A = Y_A / np.linalg.norm(Y_A)

    # Rotation axis from PCA on walking gyro
    pca = PCA(n_components=3)
    pca.fit(walk_gyro)
    Z_A = pca.components_[0, :]

    # Gram-Schmidt orthogonalization (remove Y component)
    Z_A = Z_A - (Y_A.T @ Z_A) * Y_A
    if np.linalg.norm(Z_A) < 1e-9:
        Z_A = np.array([0, 1, 0])
    Z_A = Z_A / np.linalg.norm(Z_A)

    # Force Z-axis sign
    if Z_A[1] < 0:
        Z_A = -Z_A

    # Complete coordinate system
    X_A = np.cross(Y_A, Z_A)
    X_A = X_A / np.linalg.norm(X_A)
    Z_A = np.cross(X_A, Y_A)
    Z_A = Z_A / np.linalg.norm(Z_A)

    # Sensor to Body rotation matrix
    R_S_to_B = np.column_stack([X_A, Y_A, Z_A])

    # Convert to quaternion
    r = Rotation.from_matrix(R_S_to_B)
    quat = r.as_quat()  # Returns [x, y, z, w]
    # Convert to [w, x, y, z] format
    calib_quat = Rotation.from_quat(quat)

    return calib_quat


def calculate_joint_angles(sensor_data_prox, sensor_data_dist,
                          calibration_prox, calibration_dist, seq='ZXZ'):
    """
    Calculate joint angles from two sensor quaternions

    Parameters:
    -----------
    sensor_data_prox : array (Nx4)
        Proximal sensor quaternions
    sensor_data_dist : array (Nx4)
        Distal sensor quaternions
    calibration_prox : Rotation
        Proximal calibration
    calibration_dist : Rotation
        Distal calibration
    seq : str
        Euler sequence (default 'ZXZ')

    Returns:
    --------
    roll, pitch, yaw : arrays
        Joint angles in degrees
    """
    N = sensor_data_prox.shape[0]

    # Convert to Rotation objects (MATLAB quaternion is [w,x,y,z], scipy is [x,y,z,w])
    q_S_prox = Rotation.from_quat(
        np.column_stack([sensor_data_prox[:, 1:4], sensor_data_prox[:, 0]])
    )
    q_S_dist = Rotation.from_quat(
        np.column_stack([sensor_data_dist[:, 1:4], sensor_data_dist[:, 0]])
    )

    # Apply calibration
    calibrated_prox = q_S_prox * calibration_prox
    calibrated_dist = q_S_dist * calibration_dist

    # Get Euler angles
    euler_dist = calibrated_dist.as_euler(seq, degrees=False)
    euler_prox = calibrated_prox.as_euler(seq, degrees=False)

    # Joint angles = proximal - distal
    euler_ang = euler_prox - euler_dist

    # Unwrap and convert to degrees
    roll = np.degrees(np.unwrap(euler_ang[:, 0]))
    pitch = np.degrees(np.unwrap(euler_ang[:, 1]))
    yaw = np.degrees(np.unwrap(euler_ang[:, 2]))

    return roll, pitch, yaw


def swing_detection(data, set_params):
    """
    Detect swing phases from two IMU sensors

    Parameters:
    -----------
    data : dict
        Data structure with cal field containing sensor data
    set_params : list
        [Dh, Dl, Ds, Tm, Td] threshold parameters

    Returns:
    --------
    P1, P2 : arrays
        Phase detection signals (1=swing, 0=stance)
    """
    # Process sensor 1 (index 0) and sensor 2 (index 3)
    P1, _ = process_imu(data['cal'][0], set_params)
    P2, _ = process_imu(data['cal'][3], set_params)

    return P1, P2


def process_imu(sensor, set_params):
    """
    Process single IMU sensor for swing detection

    Parameters:
    -----------
    sensor : dict
        Sensor data with quat, acc, gyr fields
    set_params : list
        [Dh, Dl, Ds, Tm, Td]

    Returns:
    --------
    P : array
        Phase detection signal
    plot_data : dict
        Intermediate signals for debugging
    """
    quat = sensor['quat']
    acc = sensor['acc']
    m = quat.shape[0]

    # Extract parameters
    Dh, Dl, Ds, Tm, Td = set_params
    gm = 0.8

    # Initialize arrays
    a = np.zeros((m, 3))
    D = np.zeros(m)
    Df = np.zeros(m)
    Df_dot = np.zeros(m)
    Hto = np.zeros(m)
    Hic = np.zeros(m)
    P = np.zeros(m)

    delta = np.zeros((5, 3))

    # State variables
    T_c = 0
    Timer = 0
    flag = 0

    for i in range(m):
        # Convert quaternion to rotation matrix
        rot_gs = quat2rot(quat[i, :])
        # Adjust acceleration (remove gravity)
        a[i, :] = (rot_gs @ acc[i, :] - np.array([0, 0, 9.8]))

        # Compute squared sum for sliding window
        sq_delta = np.sum(delta**2, axis=0)

        # Update sliding window
        delta[0:4, :] = delta[1:5, :]
        delta[4, :] = a[i, :]

        # Compute difference measure D
        D[i] = np.sum(sq_delta * np.std(delta, axis=0)) / 5

        # Exponential smoothing
        if i == 0:
            Df[i] = 0
        else:
            Df[i] = 0.7 * D[i] + 0.3 * Df[i-1]

        # Derivative of Df
        if i == 0:
            Df_dot[i] = 0
        else:
            Df_dot[i] = Df[i] - Df[i-1]

        # Swing/stance indicator
        if Df[i] > Dh:
            Hto[i] = 1
        elif Df[i] < Dl:
            Hto[i] = 0
        else:
            Hto[i] = Hto[i-1] if i > 0 else 0

        # Change indicator
        if Df_dot[i] < Ds:
            Hic[i] = 1
        elif i > 0 and (Df_dot[i] + Df_dot[i-1])/2 < gm * Ds:
            Hic[i] = 1
        else:
            Hic[i] = 0

        # Update continuous stance counter
        if Hto[i] == 1:
            T_c = 0
        else:
            T_c += 1

        # State machine for phase detection
        if Hto[i] == 1 and flag == 0:
            flag = 1
        else:
            if flag == 1:
                Timer += 1
                P[i] = 1
                if Timer > Tm and Hic[i] == 1:
                    flag = 2
                    P[i] = 0
            elif flag == 2 and Hto[i] == 1:
                Timer = 0
                P[i] = 0

        # Reset if continuous stance exceeds threshold
        if T_c > Td:
            P[i] = 0
            flag = 0
            Timer = 0
            Hto[i] = 0

    plot_data = {'Df': Df, 'Df_dot': Df_dot, 'Hto': Hto, 'Hic': Hic}
    return P, plot_data


def remove_short_pulses(signal, min_length):
    """
    Remove pulses shorter than min_length
    """
    signal = signal.flatten()
    clean_signal = signal.copy()
    binary_sig = signal > 0

    # Find edges
    edges = np.diff(np.concatenate([[0], binary_sig, [0]]))
    start_indices = np.where(edges == 1)[0]
    end_indices = np.where(edges == -1)[0] - 1

    num_pulses = len(start_indices)

    for i in range(num_pulses):
        pulse_len = end_indices[i] - start_indices[i] + 1
        if pulse_len < min_length:
            clean_signal[start_indices[i]:end_indices[i]+1] = 0

    return clean_signal


def separate_step(vector):
    """
    Separate step timing from swing signal
    Returns: step_time_data [stanceStart, stanceEnd, swingEnd]
    """
    min_length = 10
    vector = vector.flatten()

    # Find rising and falling edges
    diff_vec = np.diff(vector)
    rising_edges = np.where(diff_vec == 1)[0] + 1
    falling_edges = np.where(diff_vec == -1)[0] + 1

    if len(rising_edges) < 2 and len(falling_edges) < 2:
        return np.array([])

    # Ensure first falling edge comes before first rising edge
    if len(rising_edges) > 0 and len(falling_edges) > 0:
        if rising_edges[0] < falling_edges[0]:
            rising_edges = rising_edges[1:]
        if len(rising_edges) > 0 and len(falling_edges) > 0:
            if falling_edges[-1] > rising_edges[-1]:
                falling_edges = falling_edges[:-1]

    if len(falling_edges) == 0 or len(rising_edges) == 0:
        return np.array([])

    # Construct output matrix
    next_falling = np.concatenate([falling_edges[1:], [0]])
    step_time_data = np.column_stack([falling_edges, rising_edges, next_falling])

    # Remove last row if vector ends with falling edge or last next_falling is 0
    if vector[-1] == 0:
        step_time_data = step_time_data[:-1, :]

    if len(step_time_data) > 0 and step_time_data[-1, 2] == 0:
        step_time_data = step_time_data[:-1, :]

    # Remove invalid rows
    if len(step_time_data) > 1:
        diff_falling = np.diff(step_time_data[:, 0])
        diff_rising = np.diff(step_time_data[:, 1])
        invalid_rows = np.where((diff_falling < min_length) | (diff_rising < min_length))[0]
        step_time_data = np.delete(step_time_data, invalid_rows, axis=0)

    return step_time_data


def resampling(vector, resample_num):
    """
    Resample vector to resample_num points
    """
    re_vector = np.zeros(resample_num)
    for i in range(resample_num):
        idx = int(np.round(i * (len(vector) - 1) / (resample_num - 1)))
        re_vector[i] = vector[idx]
    return re_vector


def extracting_cyclic_data(vector, step_time_data, resample_num_st, resample_num_sw):
    """
    Extract cyclic gait data normalized to gait cycle percentage
    """
    n_steps = step_time_data.shape[0]
    cyclic_data = np.zeros((n_steps, resample_num_st + resample_num_sw))

    for i in range(n_steps):
        stance_data = vector[step_time_data[i, 0]:step_time_data[i, 1]]
        swing_data = vector[step_time_data[i, 1]:step_time_data[i, 2]]

        cyclic_st = resampling(stance_data, resample_num_st)
        cyclic_sw = resampling(swing_data, resample_num_sw)

        cyclic_data[i, :] = np.concatenate([cyclic_st, cyclic_sw])

    return cyclic_data


def extraction_gait_char(step_time_data, data):
    """
    Extract gait characteristics (max/min during stance/swing)
    """
    n_steps = step_time_data.shape[0]

    maxData_st = np.zeros(n_steps)
    maxData_sw = np.zeros(n_steps)
    minData_st = np.zeros(n_steps)
    minData_sw = np.zeros(n_steps)

    for i in range(n_steps):
        maxData_st[i] = np.max(data[step_time_data[i, 0]:step_time_data[i, 1]])
        maxData_sw[i] = np.max(data[step_time_data[i, 1]:step_time_data[i, 2]])
        minData_st[i] = np.min(data[step_time_data[i, 0]:step_time_data[i, 1]])
        minData_sw[i] = np.min(data[step_time_data[i, 1]:step_time_data[i, 2]])

    result = {
        'maxData_st': maxData_st,
        'maxData_sw': maxData_sw,
        'minData_st': minData_st,
        'minData_sw': minData_sw
    }
    return result


def remove_outliers_and_compute_mean(data, factor=1.5):
    """
    Remove outliers using IQR method and compute mean/std
    """
    data = data.flatten()
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1

    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR

    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
    mean_value = np.mean(filtered_data)
    std_value = np.std(filtered_data, ddof=1)  # Use ddof=1 for sample std

    return filtered_data, mean_value, std_value


def process_struct_fields(data_struct, factor=1.5):
    """
    Recursively process struct fields, removing outliers and computing statistics
    """
    result = {}

    for key, value in data_struct.items():
        if isinstance(value, np.ndarray):
            filtered_data, mean_value, std_value = remove_outliers_and_compute_mean(value, factor)
            result[key] = {
                'filteredData': filtered_data,
                'meanValue': mean_value,
                'stdValue': std_value
            }
        elif isinstance(value, dict):
            result[key] = process_struct_fields(value, factor)
        else:
            result[key] = value

    return result
