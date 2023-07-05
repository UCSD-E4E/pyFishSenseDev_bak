'''
    Module for reading and writing numpy arrays
'''
import numpy as np
from typing import List
import tarfile
import io

def write_np_arrays(np_array_list: List[np.ndarray]):
    pass

def _read_numpy_array(self, buffer: io.BufferedReader):
    with io.BytesIO(buffer.read()) as b:
        return np.load(b)

def read_numpy_array(file: tarfile.Tarfile, name: str) -> np.ndarray:
    return _read_numpy_array(file.extractfile(name))

def read_camera_calibration(file_path: str):
    with tarfile.open(file_path, "r:gz") as f:
        calibration_matrix = read_numpy_array(f, "calibration_matrix.npy")
        distortion_coeffs = read_numpy_array(f, "distortion_coefficients.npy")
        return calibration_matrix, distortion_coeffs

def read_laser_calibration(file_path: str): 
    pass