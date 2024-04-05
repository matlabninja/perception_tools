# This file is meant to accumulate some lidar aggregator classes that reuse some core
# functionality, but individually adapted to different datasets
import os
import numpy as np
import json
import glob
from pyquaternion import Quaternion

from .utils import read_nusc_pcd, transform_points, write_pcd

from typing import Union
from io import _io

# Class for aggregating nuscenes data. Starts with opening the dataset and collecting transforms and
# intervals. Aggregates in a later function
class NuscenesAggregator:
    def __init__(self,data_dir: str) -> None:
        '''
        Creates a NuScenes lidar aggregator. Only need the root directory of the NuScenes dataset to start
        Inputs:
            data_dir: str -> root directory of NuScenes dataset
        '''
        # Read ego pose data
        ego_pose_file = os.path.join(data_dir,'v1.0-trainval/ego_pose.json')
        with open(ego_pose_file,'r') as f:
            self.ins_data = json.load(f)
        # Read sensor table
        sensor_table = os.path.join(data_dir,'v1.0-trainval/sensor.json')
        with open(sensor_table,'r') as f:
            self.sensor_data = json.load(f)
        # Read sensor calibration table
        calib_table = os.path.join(data_dir,'v1.0-trainval/calibrated_sensor.json')
        with open(calib_table,'r') as f:
            self.calib_data = json.load(f)
        # Create a dictionary of user friendly sensor calibration keys
        self.sensorDict = {}
        for sen in self.sensor_data:
            for cd in self.calib_data:
                if cd['sensor_token'] == sen['token']:
                    sd = {}
                    sd['translation'] = cd['translation']
                    sd['rotation'] = cd['rotation']
                    sd['camera_intrinsic'] = cd['camera_intrinsic']
                    self.sensorDict[sen['channel']] = sd
        # Fetch the lidar file list
        lidar_wildcard = os.path.join(data_dir,'sweeps/LIDAR_TOP/*.pcd.bin')
        self.lidar_files = sorted(glob.glob(lidar_wildcard))
        # Extract some transforms of interest
        self.car2lidarRot = Quaternion(np.array(self.sensorDict['LIDAR_TOP']['rotation']))
        self.lidar2carRot =  self.car2lidarRot.inverse
        self.car2lidarTran = np.array(self.sensorDict['LIDAR_TOP']['translation']).reshape((3,1))
        # Extract timestamps for INS and lidar samples
        self.tsLidar = np.zeros((len(self.lidar_files),))
        for idx in range(len(self.lidar_files)):
            self.tsLidar[idx] = float(self.lidar_files[idx].split('__')[-1].split('.')[0])
        self.ins_data = sorted(self.ins_data, key = lambda x:x['timestamp'])
        self.tsIns = np.zeros((len(self.ins_data),))
        for idx in range(len(self.ins_data)):
            self.tsIns[idx] = self.ins_data[idx]['timestamp']
        # Find points where scenes begin and end
        sample_diff = np.diff(self.tsLidar)
        self.cutPoints = (np.abs(sample_diff) > 500000).nonzero()[0]+1
        self.cutPoints = np.hstack([0, self.cutPoints, len(self.tsLidar)])
        # Set output to none
        self.ptsFull = None

    def aggregate(self,segment: int) -> np.ndarray:
        '''
        Function that performs aggregation using lidar and IMU data. Outputs a numpy array but also
        stores the array to the class.
        Inputs:
          segment: int - the NuScenes dataset is set out in segments, this integer specifies which
                         one you want in time stamp order. TODO: use the build in NuScenes IDs
        Output - NxM array containing one lidar point in each row
        '''

        origin = None
        ptCloudList = []
        for idx in range(self.cutPoints[segment],self.cutPoints[segment+1]):
            # Load the lidar data
            pcd_data = read_nusc_pcd(self.lidar_files[idx])
            pts = pcd_data[0]
            # Rotate and translate to INS
            pts3carFlu = transform_points(pts[:,:3].T,self.lidar2carRot.rotation_matrix,
                                                self.car2lidarTran)
            # Get matching INS data: the nuscenes dataset has exact-match timestamps between the lidar
            # scans and the INS data
            ts_near = np.argmin(np.abs(self.tsIns-self.tsLidar[idx]))
            rotation = Quaternion(np.array(self.ins_data[ts_near]['rotation']))
            translation = np.array(self.ins_data[ts_near]['translation']).reshape((3,1))
            # Rotate to ENU
            toEnu = rotation.inverse.rotation_matrix
            # Transform points to origin of the cloud
            if origin is None:
                origin = translation
            # Tranlate to origin of cloud
            toGlobal = translation-origin
            pts3enu = transform_points(pts3carFlu,toEnu,toGlobal)
            # Reconstitute data
            ptsWrite = np.hstack([pts3enu.T,pts[:,3:]])
            ptCloudList.append(ptsWrite)
            self.ptsFull = np.vstack(ptCloudList).astype(np.float32)
        return self.ptsFull
        
    def to_pcd(self,pcd_file: Union[str,_io.BufferedReader]) -> None:
        '''
        Writes an input NxM numpy array to a pcd file specified by a binary file pointer or string.
        Does nothing if aggregation has not been run
        Inputs:
          pcd_file: _io.BufferedReader or str - binary file pointer or string file name
        '''
        # Return without doing anything if aggregation has not been run. TODO: Throw a warning
        if self.ptsFull is None:
            return
        # Set column names as needed and then write the file
        write_pcd(pcd_file,self.ptsFull,cols=['x','y','z','i','beam'])