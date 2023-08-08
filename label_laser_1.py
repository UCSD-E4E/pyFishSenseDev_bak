import cv2
import numpy as np
import os
import glob
import csv
from camera_imaging_pipeline.utils.processing_functions import imageResize
from pathlib import Path
import matplotlib.pyplot as plt
import keyboard
import json

def on_key(event):
    global point
    point = (0,0)
    return point


def getCoord(img):
    global point

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(img)
    cid = fig.canvas.mpl_connect('button_press_event', __onclick__)
    cid_key = fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()
    print(point)
    if keyboard.is_pressed('x'):
        point = (0,0)

    return np.asarray(point).astype('uint16')

def __onclick__(click):
    global point
    point = (click.xdata,click.ydata)
    return point



img_path = Path("C:/Users/Hamish/Documents/E4E/Fishsense/fishsense-lite-python-pipeline/data/7_23_La_Jolla_Kelp_Beds/Safety_Stop_Red")
files = list(img_path.glob("*.JPG"))[:20]
csv_dict = {}

for file in files:
    curr_file = file.name
    img = cv2.imread(file.as_posix())
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = imageResize(img, 25)

    coords = getCoord(img)
    print(coords)
    csv_dict[curr_file] = coords.tolist()

 
print(csv_dict)
# Serializing json
json_object = json.dumps(csv_dict, indent=4)
 
# Writing to sample.json
with open("laser_coords_1.json", "w") as outfile:
    outfile.write(json_object)

# # convert dict to list
# output_csv = []
# output_csv.append(['name','x','y'])
# for item in csv_dict.items():
#     #print(item[1][0][0])
#     output_csv.append([item[0], item[1][0][0], item[1][0][1]])

# # Write this 2d matrix into a csv file
# with open(os.fspath('./safety_stop_red.csv'), 'w') as output_file:
#     wr = csv.writer(output_file)
#     wr.writerows(output_csv)

