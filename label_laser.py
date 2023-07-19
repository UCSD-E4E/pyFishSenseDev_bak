import cv2
import numpy as np
import os
import glob
import csv
from camera_imaging_pipeline.utils.processing_functions import imageResize

#This variable we use to store the pixel location
refPt = []
curr_file = ''
img = None
img_clone = None
output_csv_dict = {}

#click event function
def click_event(event, x, y, flags, param):
    global img_clone
    global img
    global output_csv_dict
    global curr_file
    if event == cv2.EVENT_LBUTTONDOWN:
        output_csv_dict[curr_file] = [x,y]
        print(output_csv_dict[curr_file])
        font = cv2.FONT_HERSHEY_SIMPLEX
        strXY = str(x)+", "+str(y)
        img_clone = img.copy()
        cv2.putText(img_clone, strXY, (x,y), font, 0.5, (0,0,255), 2)
        cv2.imshow("image", img_clone)

 
jpg_list = glob.glob(os.fspath('data\\7_23_nathans_pool\FSL-01F_Fred\\P7130377.jpg'))
for file in jpg_list:
    curr_file = file
    img = cv2.imread(curr_file)
    img = imageResize(img, 25)
    #img_clone = img.copy()
    cv2.imshow("image", img)
    cv2.setMouseCallback("image", click_event)
    if cv2.waitKey(0) == ord('q'):
        print("skipping this image")
        output_csv_dict.pop(curr_file, None)
    cv2.destroyAllWindows()
print("all images done")

# convert dict to list
output_csv = []
output_csv.append(['name','x','y'])
for item in output_csv_dict.items():
    output_csv.append([item[0], *item[1]])

# Write this 2d matrix into a csv file
with open(os.fspath('./label_laser.csv'), 'w') as output_file:
    wr = csv.writer(output_file)
    wr.writerows(output_csv)
