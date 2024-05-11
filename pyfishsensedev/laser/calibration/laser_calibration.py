from pathlib import Path

import numpy as np

from pyfishsensedev.library.array_read_write import read_laser_calibration


class LaserCalibration:
    def __init__(self, laser_calibration_path: Path | None) -> None:
        if laser_calibration_path is not None:
            self.laser_position, self.laser_orientation = read_laser_calibration(
                laser_calibration_path.as_posix()
            )
        else:
            # Good first guess for TG6
            self.laser_position = np.array([-0.04, -0.11, 0])
            self.laser_orientation = np.array([0, 0, 1])
