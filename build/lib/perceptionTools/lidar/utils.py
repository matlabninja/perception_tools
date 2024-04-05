# This file collects some lidar utilities that are used across various lidar functions
import numpy as np
from typing import Union, Tuple, List
from io import _io

def _read_pcd_file(f: _io.BufferedReader) -> Tuple[np.ndarray,List[str]]:
    '''
    Reads a pcd file body from a binary file pointer
    Inputs:
      f: _io.BufferedReader - binary file pointer
    Outputs:
      Numpy array sized NxM with one point in each row
    '''
    # Read bytes
    buff = f.read()
    # Get the column names
    if b'FIELDS ' in buff:
        fieldSplit = buff.split(b'FIELDS ')[1].split(b'\n')[0].decode('utf-8')
        cols = fieldSplit.split(' ')
    else:
        cols = []
    # Get the data sizes, types, counts, and number of points
    sizeSplit = buff.split(b'SIZE ')[1].split(b'\n')[0].decode('utf-8')
    sizes = sizeSplit.split(' ')
    typeSplit = buff.split(b'TYPE ')[1].split(b'\n')[0].decode('utf-8')
    types = typeSplit.split(' ')
    countSplit = buff.split(b'COUNT ')[1].split(b'\n')[0].decode('utf-8')
    counts = countSplit.split(' ')
    pointSplit = buff.split(b'POINTS ')[1].split(b'\n')[0].decode('utf-8')
    points = int(pointSplit)
    # Construct datatype array for converstion
    dtypeStr = ''
    for count, size, ty in zip(counts,sizes,types):
        for n in range(int(count)):
            dtypeStr += ','+ty.lower()+size
    dtypeStr = dtypeStr[1:]
    dtypes = np.dtype(dtypeStr)
    # Construct the array
    if b'DATA binary\n' in buff:
        point_data = buff.split(b'DATA binary\n')[1]
        lidarPts = np.ndarray(points,dtypes,point_data)
        lidarPts = lidarPts.view(np.float32).reshape((points,-1))
    else:
        raise(NotImplementedError,'Text PCD not yet supported')
    return (lidarPts,cols)

def read_pcd(pcd_file: Union[str,_io.BufferedReader]) -> Tuple[np.ndarray,List[str]]:
    '''
    Reads a pcd file from a string file name or binary file pointer
    Inputs:
      pcd_file: _io.BufferedReader or str - string file name or binary file pointer
    Outputs:
      Numpy array sized NxM with one point in each row
    '''
    # Read binary data stream
    if isinstance(pcd_file,str):
        with open(pcd_file, 'rb') as f:
            pcd_data = _read_pcd_file(f)
    else:
        pcd_data = _read_pcd_file(pcd_file)

    return pcd_data

def read_nusc_pcd(pcd_file: Union[str,_io.BufferedReader]) -> Tuple[np.ndarray,List[str]]:
    '''
    Reads a nuscenes-formatted pcd file from a string file name or binary file pointer
    Inputs:
      pcd_file: _io.BufferedReader or str - string file name or binary file pointer
    Outputs:
      Numpy array sized NxM with one point in each row
    '''
    # Get data from file
    if isinstance(pcd_file,str):
        with open(pcd_file,'rb') as f:
            point_data = f.read()
    else:
        # Read bytes
        point_data = pcd_file.read()
    # Compute the number of points
    points = int(len(point_data)/20)
    # Set data type
    dtypeStr = 'f4,f4,f4,f4,f4'
    dtypes = np.dtype(dtypeStr)
    # Read the data
    lidarPts = np.ndarray(points,dtypes,point_data)
    lidarPts = lidarPts.view(np.float32).reshape((points,-1))

    return (lidarPts,['x','y','z','i','beam'])

def _write_pcd_file(f: _io.BufferedReader, header: bytes, point_data: np.ndarray, format: str) -> None:
    '''
    Writes an input NxM numpy array to a pcd file specified by a binary file pointer
    Inputs:
      f: _io.BufferedReader - binary file pointer
      header: bytes - bytes containing the header information for the pcd file
      point_data: np.ndarray - array containing the points to be written
      format: str - specifies whether the file should be written in binary or text pcd format 
    Outputs:
      Numpy array sized NxM with one point in each row
    '''
    if format == 'binary':
        f.write(header)
        f.write(b'DATA binary\n')
        f.write(point_data.tobytes())
    else:
        raise(NotImplementedError,'Text PCD not yet supported')

def write_pcd(pcd_file: Union[str,_io.BufferedReader],point_data: np.ndarray, cols: List[str] = [],\
              sizes: Union[str,np.array,List[str],List[int]] = [],types: Union[str,List[str]] = [], \
              counts: Union[str,np.array,List[str],List[int]] = [], format: str = 'binary') -> None:
    '''
    Writes an input NxM numpy array to a pcd file specified by a binary file pointer or string
    Inputs:
      pcd_file: _io.BufferedReader or str - binary file pointer or string file name
      point_data: np.ndarray - array containing the points to be written
      cols: list - list of strings containing column names for the point array
             defaults to not writing column names
      sizes: list or str - list or str containing size in bytes for entries in each column
             defaults to 4 bytes per value
      types: list or str - list or str containing the data type for each column
             defaults to floating point
      counts: list or str - list or str containing the number of elements in each row of each column
              defaults to one per row per column
      format: str - specifies whether the file should be written in binary or text pcd format 
    Outputs:
      Numpy array sized NxM with one point in each row
    '''
    # Start creating header
    header = b'VERSION .7\n'
    if len(cols) > 0:
        fields = b'FIELDS'
        for field in cols:
            nextBit = b' ' + field.encode('utf-8')
            fields = fields + nextBit
        header = header + fields + b'\n'
    size = b'SIZE'
    if len(sizes) > 0:
        for sz in sizes:
            nextBit = b' ' + str(sz).encode('utf-8')
            size = size + nextBit
    else:
        for _ in range(point_data.shape[1]):
            size = size + b' 4'
    header = header + size + b'\n'
    dtype = b'TYPE'
    if len(types) > 0:
        for ty in types:
            nextBit = b' ' + str(ty).encode('utf-8')
            dtype = dtype + nextBit
    else:
        for _ in range(point_data.shape[1]):
            dtype = dtype + b' F'
    header = header + dtype + b'\n'
    count = b'COUNT'
    if len(counts) > 0:
        for ct in counts:
            nextBit = b' ' + str(ct).encode('utf-8')
            count = count + nextBit
    else:
        for _ in range(point_data.shape[1]):
            count = count + b' 1'
    header = header + count + b'\n'
    w = b'WIDTH '+bytes(str(point_data.shape[0]),'utf-8')
    header = header + w + b'\n'
    h = b'HEIGHT 1'
    header = header + h + b'\n'
    vp = b'VIEWPOINT 0 0 0 1 0 0 0'
    header = header + vp + b'\n'
    pts = b'POINTS '+bytes(str(point_data.shape[0]),'utf-8')
    header = header + pts + b'\n'
    # With header constructed, create file
    if isinstance(pcd_file,str):
        if format == 'binary':
            with open(pcd_file, 'wb') as f:
                _write_pcd_file(f,header,point_data,format)
        else:
            raise(NotImplementedError,'Text PCD not yet supported')
    else:
        _write_pcd_file(pcd_file,header,point_data,format)

def transform_points(pts: np.ndarray, rot: np.ndarray, t: np.ndarray) -> np.ndarray:
    '''
    Applies a rotation and translation to a set of points
    Inputs:
      pts: np.ndarray - 3xN array of points to transform, one point per column
      rot: np.ndarray - 3x3 rotation matrix
      t: np.ndarray - 3x1 translation vector
    '''
    # Rotate
    pts = rot@pts
    # Translate
    return pts - t