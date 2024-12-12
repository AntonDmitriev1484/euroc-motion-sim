from scipy import integrate

import pypose as pp
import torch

from load_data import *

# ~/Datasets/EuRoC_orbslam3_data/drone_imu/V101_imu0

# Velocity radians ... acceleration
T, GX, GY, GZ, AX, AY, AZ = 0, 1, 2, 3, 4, 5, 6

# The evo TUM conversion converts to seconds I think.
# So groundtruth timestamps are in seconds (decimal)

# print(f" GT length {gt_traj.shape[0]}")
# print(f" IMU length {imu_raw.shape[0]}")
def check_dT(gt_traj, imu_raw):
    print("For ground_truth")
    gt_traj_diff = []

    for i in range(gt_traj.shape[0]-1):
        gt_traj_diff.append(abs(gt_traj[i]*(10**9) - gt_traj[i+1]*(10**9)))

    print(f"Average timestamp difference (ns) : {np.mean(np.array(gt_traj_diff)[:,0])}")
    print("For IMU data")
    imu_raw_diff = []

    for i in range(imu_raw.shape[0]-1):
        imu_raw_diff.append(abs(imu_raw[i] - imu_raw[i+1]))

    print(f"Average timestamp difference (ns) {np.mean(np.array(imu_raw_diff)[:,0])}")


#  GT length 28711
#  IMU length 29119
# For ground_truth
# Average timestamp difference (ns) : 5000000.004458377
# For IMU data
# Average timestamp difference (ns) 5000000.004395906
# So average delta time between two rows is 5ms


def skew_symmetric(omega):
    return np.array([
        [0, -omega[2], omega[1]],
        [omega[2], 0, -omega[0]],
        [-omega[1], omega[0], 0]
    ])

# I still don't quite understand why we are using the skew symmetric here
# When nothing online uses it
def exp_rodrigues(omega, dt):
    
    theta = np.linalg.norm(omega) * dt
    cross_omega = skew_symmetric(omega)

    if np.linalg.norm(omega) < 1e-6: # For small theta formula simplifies
        return np.identity(3) + cross_omega
    
    return (np.identity(3) 
            + cross_omega*np.sin(theta)/np.linalg.norm(omega) 
            + np.dot(cross_omega, cross_omega)*(1-np.cos(theta))/(np.linalg.norm(omega)**2))

def integrate_imu(gt_traj, imu_raw):

    # dT = 0.005
    dT = 0

    # TODO: Do they put their vertical as Y or Z?
    T, wX, wY, wZ, aX, aY, aZ = 0, 1, 2, 3, 4, 5, 6 # Indices for IMU raw data
    T, X, Y, Z, RO, PI, YA = 0, 1, 2, 3, 4, 5, 6    # Indices for trajectory

    imu_traj = np.zeros(gt_traj.shape)
    imu_traj[0, :] = gt_traj[0, :]  # Initialize with ground truth start point
    imu_traj[0, T] = 0

    # Assuming that initially the IMu starts perfectly aligned with global frame.
    R_ig = np.identity(3)
    v_g = np.zeros(3)
    p_g = np.zeros(3)

    for t in range(1, gt_traj.shape[0]):

        # I don't think its the rodriguez formula thats messing it up?
        
        d_R_ig = exp_rodrigues(imu_raw[t,[wX,wY,wZ]] , dT)
        # Gives us the change in our rotation matrix
        R_ig = np.matmul(R_ig , d_R_ig)

        a_i = imu_raw[t,[aX,aY,aZ]]
        # a_g = np.matmul(R_ig, a_i)
        a_g = np.matmul(R_ig, a_i) + np.array((0,0,-9.81)) # Compensate for acceleration down z axis in global frame

        v_g += dT * a_g
        p_g += dT * v_g

        timestamp = imu_raw[t,T]

        # imu_traj[t, :] = np.array([timestamp, p_g[0], p_g[1], p_g[2], 0, 0, 0])
        imu_traj[t, T] = t * dT
        imu_traj[t, X] = p_g[0]
        imu_traj[t, Y] = p_g[1]
        imu_traj[t, Z] = p_g[2]
        imu_traj[t, RO] = 0
        imu_traj[t, PI] = 0
        imu_traj[t, YA] = 0

    return imu_traj




gt_traj = read_tum_as_pose("/home/admitriev/Datasets/EuRoC_orbslam3_data/ground_truth/V101_state_groundtruth_estimate0/data.tum")
imu_raw = read_standard("/home/admitriev/Datasets/EuRoC_orbslam3_data/drone_imu/V101_imu0/data.csv")
print(gt_traj.shape)

imu_traj = integrate_imu(gt_traj, imu_raw)

write_pose_to_tum("/home/admitriev/ISF24/euroc-motion-sim/drone1_gt.tum", gt_traj[:500])
write_pose_to_tum("/home/admitriev/ISF24/euroc-motion-sim/drone1_imu.tum", imu_traj[:500])

# TODO: Write integrated IMU to suitable output file
# Then compare xyz view of trajectory to see the drift
# Might also truncate both datasets to like 400 or 500 timestamps to confirm drift at the start


    