# Fishsense Lite Python Pipeline
This repository contains a Python implementation of the Fishsense lite pipeline. It contains tools for camera calibration, laser calibration, and fish length calculation. This pipeline assumes that you already have access to all the data necessary to run this pipeline. As this code is meant to be a temporary measure, the user may need to edit some file paths, so apologies in advance. 

## Steps to use
1. Run `poetry install`. This will install all required dependencies.
2. Run `poetry shell` to execute all following commands within the poetry environment.
3. Create a `data` folder in this directory. Add a `raw` subdirectory and add your raw data inside. This can take whatever folder structure you need. 
4. Process all raws by running `python camera_imaging_pipeline/process.py`. This will place all images inside `data/png`, copying whatever folder structure is inside `data/raw`.
4. Calibrate the lens using `python e4e_camera_calibration`. Documentation on this tool can be found [here](https://github.com/UCSD-E4E/e4e-camera-calibration).
5. Undistort all laser and fish images using `python undistort_images.py`. This will place all images inside `data/png_rectified`, copying whatever folder structure is inside `data/png`.
6. Label laser calibration images using `python label_laser_calibration.py`. 
     * This script uses the following command line arguments: 
        * `-i` or `--input` specifies the input path which contains all *rectified* laser images.
        * `-o` or `--output` specifies the location to save the generated csv file.
     * Use the following key bindings when using this tool: 
        * `ESC` will skip the current image, use this when the laser dot is difficult to find, the calibration board is not completely in frame, or things are out of focus.
        * `r` will remove the current label, in case you make a mistake
        * `e` will save the added label and load the next image
7. Calibrate the laser using `python laser_calibration.py`. This script uses the following command line arguments:
     * `-c` or `--calib` specifies the location of the lens calibration file generated from step 3. 
     * `-i` or `--input` specifies the location of the csv file from part 5. 
     * `-o` or `--output` specifies the location to save the generated laser calibration file.
8. Label the fish images using `python label_laser_fish.py`
     * This script uses the following command line arguments: 
        * `-i` or `--input` specifies the input path which contains all *rectified* fish with laser images.
        * `-o` or `--output` specifies the location to save the generated csv file.
     * Click to label the laser dot, snout tip, and tail fork, in that order. 
     * Use the following key bindings when using this tool: 
        * `ESC` will skip the current image, use this when the laser dot is difficult to find, or things are out of focus.
        * `r` will remove all current labels, in case you make a mistake
        * `e` will save the added label and load the next image
9. Run the fish analysis using `python fish_mass.py`. This script uses the following command line arguments: 
     * `-c` or `--cameracalib` specifies the location of the lens calibration file generated from step 3. 
     * `-l` or `--lasercalib` specifies the location of the laser calibration file generated from step 6.
     * `-i` or `--input` specifies the location of the csv file generated from step 7.
     * `-o` or `--output` specifies the location of the generated csv file containing fish lengths and masses. 
