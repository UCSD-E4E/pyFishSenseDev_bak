import csv
import numpy as np
from typing import Tuple, List
from laser_parallax import compute_world_points, compute_world_points_from_depths
from array_read_write import read_camera_calibration, read_laser_calibration
from constants import *
import matplotlib.pyplot as plt

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

# Calculate fish length using this
def get_fish_lengths(snout_3d: np.ndarray, 
                     fork_3d: np.ndarray) -> np.ndarray:
    # Calculate length of fish using linalg norm
    return np.linalg.norm(snout_3d-fork_3d, axis=1)

# Calculate fish mass using species and fish length
def get_fish_masses(fish_lengths: np.ndarray, species: List[str]) -> np.ndarray:

    # Temporary return
    return np.zeros(fish_lengths.shape)

def main():
    
    # Get the map made from the csv file

    file_map = read_csv("fish_data.csv")

    # Get the laser position and orientation from the laser calibration file

    laser_position, laser_orientation = read_laser_calibration("laser-calibration-output-4-12.dat")
    camera_mat, _ = read_camera_calibration('calibration-output.dat')
    focal_length_mm = camera_mat[0][0] * PIXEL_PITCH_MM
    sensor_size_px = np.array([4000,3000])
    camera_params = (focal_length_mm, sensor_size_px[0], sensor_size_px[1], PIXEL_PITCH_MM)


    # Prepare to write to the output csv file by creating a 2d array
    output_csv = []
    output_csv.append(["File_Name", "Length", "Mass"])
    
    laser_coords = np.zeros((len(file_map),2))
    snout_coords = np.zeros((len(file_map),2))
    fork_coords = np.zeros((len(file_map),2))
    species_list = []
    for i, file in enumerate(file_map):

        # Read values from map

        laser_x = int(file_map[file]["laser.x"])
        laser_y = int(file_map[file]["laser.y"])
        snout_x = int(file_map[file]["head.x"])
        snout_y = int(file_map[file]["head.y"])
        fork_x = int(file_map[file]["tail.x"])
        fork_y = int(file_map[file]["tail.y"])
        # species = int(file_map[file]["species"])

        laser_coords[i] = [laser_x, laser_y]
        snout_coords[i] = [snout_x, snout_y]
        fork_coords[i] = [fork_x, fork_y]
        # species_list.append(species)

    laser_3d_coords = compute_world_points(
        laser_origin=laser_position, 
        laser_axis=laser_orientation,
        camera_params=camera_params, 
        image_coordinates=(sensor_size_px/2 - laser_coords)
    )

    depths = laser_3d_coords[:,2]
    snout_3d_coords = compute_world_points_from_depths(
        camera_params=camera_params, 
        image_coordinates=(sensor_size_px/2 - snout_coords), 
        depths=depths
    )
    fork_3d_coords = compute_world_points_from_depths(
        camera_params=camera_params, 
        image_coordinates=(sensor_size_px/2 - fork_coords), 
        depths=depths
    )

    fish_lengths = get_fish_lengths(snout_3d_coords, fork_3d_coords)

    masses = get_fish_masses(fish_lengths, species_list)

    # reference = np.zeros(fish_lengths.shape)
    # reference[:] = 0.31
    # plt.plot(depths, fish_lengths, '.', label='Estimated fish lengths')
    # plt.plot(depths, reference, '.', label='Actual fish lengths')
    # plt.xlabel('Distance from camera (m)')
    # plt.ylabel('Measured fish length (m)')
    # plt.grid()
    # plt.ylim(bottom=0, top=0.4)
    # plt.legend()
    # plt.show()
    
    for i,file in enumerate(file_map):
        # output_csv.append(file, fish_lengths[i], masses[i], file_map[file]["species"])
        output_csv.append([file, fish_lengths[i], masses[i]])

    # Write this 2d matrix into a csv file
    with open('./output.csv', 'w') as output_file:
        
        wr = csv.writer(output_file)
        wr.writerows(output_csv)

if __name__ == "__main__":
    main()

    


