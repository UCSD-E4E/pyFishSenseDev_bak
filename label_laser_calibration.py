import cv2
import numpy as np
import os
import glob
import csv
from copy import deepcopy
from laser_dot_guess_correction import correct_laser_dot

#This variable we use to store the pixel location
refPt = []
curr_file = ''
img = None
img_clone = None
output_csv_dict = {}
count = 0
glob_list = []

#click event function
def click_event(event, x, y, flags, param):
    global img_clone
    global img
    global output_csv_dict
    global curr_file
    global count
    if event == cv2.EVENT_LBUTTONDOWN:
        

        font = cv2.FONT_HERSHEY_SIMPLEX
        strXY = str(x)+", "+str(y)

        if len(glob_list) == 0:
            img_clone = img.copy()
            strLaser = "Laser: " + strXY
            newX,newY = correct_laser_dot(coord=np.array([x,y]),img=img)
            glob_list.append((newX,newY))
            cv2.putText(img_clone, strLaser, (int(newX),int(newY)), font, 0.5, (0,255,0), 2)
            cv2.circle(img_clone, (int(newX), int(newY)), radius=3, color=(0,0,255), thickness=-1)
            
        cv2.imshow("Resized_Window", img_clone)

 
jpg_list = glob.glob(os.fspath('data/laser_jpgs_rectified/*.JPG'))
for file in jpg_list:
    curr_file = file
    img = cv2.imread(curr_file)
    img_clone = img.copy()

    cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)
    cv2.imshow("Resized_Window", img)
    cv2.setMouseCallback("Resized_Window", click_event)
    while True:

        k = cv2.waitKey(0) & 0xFF

        # Add nothing to the csv if click 'esc'
        if k == 27:
            print("skipping this image")
            glob_list.clear()
            cv2.destroyAllWindows()
            break
        
        # If want to delete current values and reannotate image, click 'r'
        if k == ord('r'):
            print("resetting on current image")
            glob_list.clear()
        
        # If want to finish and move onto the next image, click 'e'
        if k == ord('e'):

            if len(glob_list) == 0:
                print("\n\nPlease redo annotations. You have reset and not recompleted.")
                exit()
            print("Image annotation complete.")
            output_csv_dict[curr_file] = deepcopy(glob_list)
            print(f"File Name: {curr_file}")
            print(f"Laser Position: {glob_list[0]}")
            glob_list.clear()
            cv2.destroyAllWindows()
            break

    
print("all images done")

# convert dict to list
output_csv = []
output_csv.append(['name','laser.x','laser.y'])
for item in output_csv_dict.items():
    output_csv.append([item[0], *item[1][0]])

# Write this 2d matrix into a csv file
with open(os.fspath('./laser_data_4_12_bot_float.csv'), 'w') as output_file:
    wr = csv.writer(output_file)
    wr.writerows(output_csv)
