import cv2
import numpy as np
import os
import glob
import csv
from copy import deepcopy
from laser_dot_guess_correction import correct_laser_dot
import argparse

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
            strLaser = "Point 1: " + strXY
            newX, newY = (x,y)
            glob_list.append((newX,newY))
            cv2.putText(img_clone, strLaser, (int(newX),int(newY)), font, 0.5, (0,255,0), 2)
            cv2.circle(img_clone, (int(newX),int(newY)), radius=3, color=(0,0,255), thickness=-1)
            
        
        elif len(glob_list) == 1:
            strHead = "Point 2: " + strXY
            glob_list.append((x,y))
            cv2.putText(img_clone, strHead, (x,y), font, 0.5, (0,255,0), 2)
            cv2.circle(img_clone, (x,y), radius=3, color=(0,0,255), thickness=-1)
            
        elif len(glob_list) == 2:
            strTail = "Point 3: " + strXY
            glob_list.append((x,y))
            cv2.putText(img_clone, strTail, (x,y), font, 0.5, (0,255,0), 2)
            cv2.circle(img_clone, (x,y), radius=3, color=(0,0,255), thickness=-1)
        
        elif len(glob_list) == 3:
            strTail = "Point 4: " + strXY
            glob_list.append((x,y))
            cv2.putText(img_clone, strTail, (x,y), font, 0.5, (0,255,0), 2)
            cv2.circle(img_clone, (x,y), radius=3, color=(0,0,255), thickness=-1)
        
        elif len(glob_list) == 4:
            strTail = "Point 5: " + strXY
            glob_list.append((x,y))
            cv2.putText(img_clone, strTail, (x,y), font, 0.5, (0,255,0), 2)
            cv2.circle(img_clone, (x,y), radius=3, color=(0,0,255), thickness=-1)
        
        elif len(glob_list) == 5:
            strTail = "Point 6: " + strXY
            glob_list.append((x,y))
            cv2.putText(img_clone, strTail, (x,y), font, 0.5, (0,255,0), 2)
            cv2.circle(img_clone, (x,y), radius=3, color=(0,0,255), thickness=-1)
        
        elif len(glob_list) == 6:
            strTail = "Point 7: " + strXY
            glob_list.append((x,y))
            cv2.putText(img_clone, strTail, (x,y), font, 0.5, (0,255,0), 2)
            cv2.circle(img_clone, (x,y), radius=3, color=(0,0,255), thickness=-1)
        
        elif len(glob_list) == 7:
            strTail = "Point 8: " + strXY
            glob_list.append((x,y))
            cv2.putText(img_clone, strTail, (x,y), font, 0.5, (0,255,0), 2)
            cv2.circle(img_clone, (x,y), radius=3, color=(0,0,255), thickness=-1)
        
        cv2.imshow("Resized_Window", img_clone)

def prep_args():
    parser = argparse.ArgumentParser(prog='label_laser_calibration',
                                     description='Labeling tool for laser calibration images')
    parser.add_argument('-i', '--input', help='Folder containing all images to be labeled', dest='input_path', required=True)
    parser.add_argument('-o', '--output', help='CSV containing image paths and point coordinates', dest='output_csv', required=True)
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = prep_args() 
    jpg_list = glob.glob(os.path.join(args.input_path,'*.PNG'))
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
                    print("Please redo annotations. You have reset and not recompleted.")
                    continue
                if len(glob_list) < 8:
                    print("Please finish annotations")
                    continue
                print("Image annotation complete.")
                output_csv_dict[curr_file] = deepcopy(glob_list)
                print(f"File Name: {curr_file}")
                print(f"Point 1: {glob_list[0]}")
                print(f"Point 2: {glob_list[1]}")
                print(f"Point 3: {glob_list[2]}")
                print(f"Point 4: {glob_list[3]}")
                print(f"Point 5: {glob_list[4]}")
                print(f"Point 6: {glob_list[5]}")
                print(f"Point 7: {glob_list[6]}")
                print(f"Point 8: {glob_list[7]}")

                glob_list.clear()
                cv2.destroyAllWindows()
                break

        
    print("all images done")

    # convert dict to list
    output_csv = []
    output_csv.append(['name','Point1X','Point1Y' ,'Point2X','Point2Y','Point3X','Point3Y','Point4X','Point4Y', 'Point5X','Point5Y',  'Point6X','Point6Y',  'Point7X','Point7Y',  'Point8X','Point8Y'])
    for item in output_csv_dict.items():
        output_csv.append([item[0], *item[1][0], *item[1][1], *item[1][2],*item[1][3],*item[1][4],*item[1][5],*item[1][6],*item[1][7]])

    # Write this 2d matrix into a csv file
    with open(args.output_csv, 'w') as output_file:
        wr = csv.writer(output_file)
        wr.writerows(output_csv)
