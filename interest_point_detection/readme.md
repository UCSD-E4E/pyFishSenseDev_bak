# Interest Point Detection for the FishSense project.
We explored interest point detection as way to detect a laser within an image. For this, we applied the SIFT algorithm, which returns a list of keypoints that it detects to be a significant feature. Since the laser is a somewhat unique point within the underwater images, the laser was detected in most cases when applying the SIFT algorithm. We then looked through the list of keypoints returned by the SIFT algorithm, and looked for the keypoint with the highest mean red value in it's proximity. 


## Using the Interest Point Detection Code
You can find an example of how the interest point detection code works in the detect_laser.py file, under the display_interest_points.py function. 
We first initialize the SIFT algorithm in line 256, by passing it the number of features it should detect, the number of octave layers, the contrat threshold, the edge threshold, and the sigma value. Refer to the opencv docs for more information: 

- https://docs.opencv.org/3.4/d7/d60/classcv_1_1SIFT.html
- https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html

Then we apply the SIFT algorithm to the image in line 257. After that, we look for the keypoint with the highest mean red value in it's proximity. We finally draw the keypoint onto the image as well as a white background, and display both images. 