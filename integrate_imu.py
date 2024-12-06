from scipy import integrate

import pypose as pp
import torch

from load_data import *

# ~/Datasets/EuRoC_orbslam3_data/drone_imu/V101_imu0

# Velocity radians ... acceleration
T, GX, GY, GZ, AX, AY, AZ = 0, 1, 2, 3, 4, 5, 6


def integrate_imu_bad():
    integrated_data = []

    for i in range(1,4):
        path = (f'/home/admitriev/Datasets/EuRoC_orbslam3_data/drone_imu/V10{i}_imu0/data.csv')
        nparr = read_standard(path)

        # -2 because we have to take 2 antiderivatives for acceleration
        integrated = np.zeros((nparr.shape[0]-2, nparr.shape[1]))
        integrated[:,T] = nparr[:-2,T]

        dt = (nparr[:,T])/(10**(9)) # Should be difference in seconds between each entry
        dt = dt - dt[0] # Normalize to start with 0
        print(f"diff in seconds = {dt[-1] - dt[0]}") # Ok based off of this dt is correct
        print(dt)
        # Why is the final timestamp value 1 or 8, this should be in seconds?

        print(nparr.shape)
        print(dt.shape)
        # Accumulated error? -> This is a pain, maybe use the kitti imu integrator

        for field in [AX, AY, AZ]:
            first_antiderivative = integrate.cumtrapz(nparr[:,field], x=dt) # will be len 1 less than original
            second_antiderivative = integrate.cumtrapz(first_antiderivative, x=dt[:-1]) # -1 array length per antiderivative
            integrated[:,field] = second_antiderivative

        for field in [GX, GY, GZ]:
            first_antiderivative = integrate.cumtrapz(nparr[:,field], x=dt)
            integrated[:,field] = first_antiderivative[:-1]

        integrated_data.append(integrated)

    return integrated_data

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

# Need to orient my x in the same direction as initial velocity

def integrate_imu_bad2(gt_traj, imu_raw):



    # dT = 0.005 # I think dT might be wrong? But doing it the same way as mrclam.
    dT = 0.005

    imu_traj = np.zeros(gt_traj.shape)
    imu_traj[0, :] = gt_traj[0, :]  # Initialize with ground truth start point

    # Indices for IMU raw data
    T, wX, wY, wZ, aX, aY, aZ = 0, 1, 2, 3, 4, 5, 6
    # Indices for trajectory
    T, X, Y, Z, RO, PI, YA = 0, 1, 2, 3, 4, 5, 6

    # Initialize velocity
    vX, vY, vZ = 0, 0, 0

    for i in range(1, gt_traj.shape[0]):

        prev_pose = imu_traj[i - 1]
        new_pose = np.zeros((7))

        # Maybe need to make an acceleration vector, and rotate according to wx,wy,wz vector to get a velocity vector
        # Update velocities using accelerations

        # Integrating acceleration gives us instantaneous velocity
        vX += imu_raw[i, aX] * dT # This is because acceleration is never negative?
        vY += imu_raw[i, aY] * dT # Acceleration never decreasing, means vX increases to insane amount, means dx is insanely high each time
        vZ += imu_raw[i, aZ] * dT
        # Ok changing this to not be a sum seemed to fix it
        # Why do we still have the giant leap in position at the start?
        

        # Adjust displacements using orientation
        roll, pitch, yaw = prev_pose[RO], prev_pose[PI], prev_pose[YA]

        # Integrating instantaneous velocity gives us change in position
        dx = vX * dT #dX is growing rapidly, never decerasing, this is because acceleration is never negative?
        dy = vY * dT
        dz = vZ * dT

        print(f"dx:{dx} dy:{dy} dz:{dz}")

        new_pose[X] = prev_pose[X] + (
            np.cos(yaw) * np.cos(pitch) * dx
            + (np.cos(yaw) * np.sin(pitch) * np.sin(roll) - np.sin(yaw) * np.cos(roll)) * dy
            + (np.cos(yaw) * np.sin(pitch) * np.cos(roll) + np.sin(yaw) * np.sin(roll)) * dz
        )
        new_pose[Y] = prev_pose[Y] + (
            np.sin(yaw) * np.cos(pitch) * dx
            + (np.sin(yaw) * np.sin(pitch) * np.sin(roll) + np.cos(yaw) * np.cos(roll)) * dy
            + (np.sin(yaw) * np.sin(pitch) * np.cos(roll) - np.cos(yaw) * np.sin(roll)) * dz
        )
        new_pose[Z] = prev_pose[Z] + (
            -np.sin(pitch) * dx
            + np.cos(pitch) * np.sin(roll) * dy
            + np.cos(pitch) * np.cos(roll) * dz
        )

        # Update orientation (roll, pitch, yaw) by integrating angular velocities
        new_pose[RO] = prev_pose[RO] + imu_raw[i, wX] * dT
        new_pose[PI] = prev_pose[PI] + imu_raw[i, wY] * dT
        new_pose[YA] = prev_pose[YA] + imu_raw[i, wZ] * dT

        # Normalize angles to [-pi, pi]
        # new_pose[RO] = (new_pose[RO] + np.pi) % (2 * np.pi) - np.pi
        # new_pose[PI] = (new_pose[PI] + np.pi) % (2 * np.pi) - np.pi
        # new_pose[YA] = (new_pose[YA] + np.pi) % (2 * np.pi) - np.pi

        # Update time
        new_pose[T] = imu_raw[i, T]/(10**9)

        imu_traj[i] = new_pose

    return imu_traj


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

    if np.linalg.norm(omega) < 1e-6: # Avoid division by zero return np.identity(3)
        return np.identity(3)
    
    return (np.identity(3) 
            + cross_omega*np.sin(theta)/np.linalg.norm(omega) 
            + np.dot(cross_omega, cross_omega)*(1-np.cos(theta))/(np.linalg.norm(omega)**2))

def integrate_imu(gt_traj, imu_raw):

    dT = 0.005

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


    