
import pandas as pd
from io import FileIO
from collections import namedtuple
import math as math
import csv
import numpy as np
from scipy.spatial.transform import Rotation as R

T, X,Y,Z, RO,P,Y = 0, 1, 2, 3, 4, 5, 6

# TUM format
# timestamp x y z q_x q_y q_z q_w

def read_standard(file):
    fstream = FileIO(file)
    df = pd.read_csv(fstream, delimiter=',', comment='#')
    arr = df.to_numpy()
    return arr

def read_tum_as_pose(file):
    df = pd.read_csv(file, delimiter=' ', comment='#', float_precision='round_trip')
    arr = df.to_numpy()
    pose_arr = np.zeros((arr.shape[0], arr.shape[1]-1))
    pose_arr[:,:4] = arr[:,:4] # t,x,y,z stays same
    pose_arr[:,4:] = R.from_quat(arr[:,4:]).as_euler('xyz')
    return pose_arr

def write_pose_to_tum(filename, nparr): 
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=' ')
        file.write("# timestamp tx ty tz qx qy qz qw\n") 
        for row in nparr:
            quat = R.from_euler('xyz', [row[RO], row[P], row[Y]]).as_quat()
            a = [ row[0], row[1], row[2], row[3], quat[0], quat[1], quat[2], quat[3]] # Don't trust the literals
            formatted = [f"{field}" for field in a]
            writer.writerow(formatted)
