
from integrate_imu import *
from load_data import *
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# TODO: First, visualize to make sure imu integration is correct

# Velocity radians ... acceleration
T, X,Y,Z, RO,P,Y = 0, 1, 2, 3, 4, 5, 6

imu_pose = integrate_imu()[0] # Want to get imu integration for first dataset
gt_pose = read_tum_as_pose("/home/admitriev/Datasets/EuRoC_orbslam3_data/ground_truth/V101_state_groundtruth_estimate0/data.tum")
#Note: Don't even know if these are timestamp aligned, imu has some 7k less poses than GT
# probabily will need to do some interpolation

print(imu_pose.shape)
print(gt_pose.shape)

print(imu_pose[5000:5010]) # These values are huge e^18 why?

end = min(imu_pose.shape[0], gt_pose.shape[0])

# Yeah one or both of these are definitely wrong lol

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(gt_pose[:end,X], gt_pose[:end,Y], gt_pose[:end,Z], c='green')
# plt.show()

ax.scatter(imu_pose[:end,X], imu_pose[:end,Y], imu_pose[:end,Z], c='red')

plt.show()
# plt.scatter(gt_pose[:end,X], gt_pose[:end,Y], gt_pose[:end,Z], c='green')
# plt.scatter(imu_pose[:end,X], imu_pose[:end,Y], imu_pose[:end,Z], c='red')