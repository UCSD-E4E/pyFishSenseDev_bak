import csv
import numpy as np
from typing import Tuple

# Function to take in file name and create a map from file name to
# The key information in the file

def read_csv(csv_file: str) -> dict:

    file_map = dict()

    # Open up the CSV and begin storing values in map

    with open(csv_file, newline='') as csvfile:

        reader = csv.reader(csvfile, delimiter=',')

        header = next(reader)

        for row in reader:
            
            file_map[row[0]] = dict()
            
            for i in range(1, len(header)):
                file_map[row[0]][header[i]] = row[i]
    
    return file_map

# Generic 3d estimation function
# TODO
def estimation_3d(laser_position: np.ndarray, 
                        laser_orientation: np.ndarray, 
                        x: float, y: float) -> np.ndarray:
    
    # Temporary Return
    return np.zeros(0)

# Overloaded function for 3d estimation. It uses the laser's 3d value to calculate other points
# TODO
def estimation_3d_with_laser(laser_3d_estimate: np.ndarray, 
                             x: int, y:int) -> np.ndarray:

    # Temporary return
    return np.zeros(0)


# Returns the laser 3d estimation given 
def laser_3d_estimation(laser_position: np.ndarray, 
                        laser_orientation: np.ndarray, 
                        laser_x: float, laser_y: float) -> np.ndarray:
    
    return estimation_3d(laser_position, laser_orientation, laser_x, laser_y)

# Gets the laser calibration parameters from the laser calibration file 
# TODO
def laser_calibration_params(calibration_file: str) -> Tuple[np.ndarray, np.ndarray]:

    # Temporary Return. 
    laser_position = np.zeros(0)
    laser_orientation = np.zeros(0)
    return laser_position, laser_orientation

# Estimates the position of the snout and fork in 3d space
def snout_fork_3d_estimation(laser_3d_estimate, 
                        snout_x: int, snout_y: int,
                        fork_x: int, fork_y: int) -> Tuple[np.ndarray, np.ndarray]:
    
    snout_3d = estimation_3d_with_laser(laser_3d_estimate, snout_x, snout_y)
    fork_3d = estimation_3d_with_laser(laser_3d_estimate, fork_x, fork_y)

    return snout_3d, fork_3d

# Calculate fish length using this

def fish_length(snout_3d: np.ndarray, 
                     fork_3d: np.ndarray) -> float:
    
    # Calculate length of fish using linalg norm

    return np.linalg.norm(snout_3d-fork_3d)

# Calculate fish mass using species and fish length

def fish_mass(fish_length: float, species: str) -> float:

    # Temporary return
    return 0

def main():
    
    # Get the map made from the csv file

    file_map = read_csv("test.csv")

    # Get the laser position and orientation from the laser calibration file

    laser_position, laser_orientation = laser_calibration_params("calibration_file.smth")

    # Prepare to write to the output csv file by creating a 2d array
    output_csv = [[]]
    output_csv.append(["File_Name", "Length", "Mass", "Species"])
    
    for file in file_map:

        # Read values from map

        laser_x = int(file_map[file]["laser.x"])
        laser_y = int(file_map[file]["laser.y"])
        snout_x = int(file_map[file]["snout.x"])
        snout_y = int(file_map[file]["snout.y"])
        fork_x = int(file_map[file]["fork.x"])
        fork_y = int(file_map[file]["fork.y"])
        species = int(file_map[file]["species"])

        # Storing the laser point's location in 3d space
        laser_3d = laser_3d_estimation(laser_position, laser_orientation,
                                        laser_x, laser_y)
        
        # Calculate the snout and fork positions in 3d space
        snout_3d, fork_3d = snout_fork_3d_estimation(laser_3d,
                                                       snout_x, snout_y, fork_x, fork_y)
        
        # Calculate the fish length using the snout and fork positions
        fish_len = fish_length(snout_3d, fork_3d)

        # Calculate the fish mass using the length and species
        mass = fish_mass(fish_len, species)

        output_csv.append([file, fish_len, mass, species])
        
    # Write this 2d matrix into a csv file

    with open('./output.csv', 'wb') as output_file:
        
        wr = csv.writer(output_file)
        wr.writerows(output_csv)

if __name__ == "__main__":
    main()

    


