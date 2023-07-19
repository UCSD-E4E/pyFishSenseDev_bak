from camera_imaging_pipeline.src.image_processing import imageProcessing
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import LightSource
from camera_imaging_pipeline.utils.processing_functions import imageResize

#read a sample image, read the parametrs .json file and load the parameters as a dictionary
#img1_path = os.path.join(os.path.dirname(__file__), '.\data\\test_data\Fishsense.dng')

img1_path = os.path.join(os.path.dirname(__file__), r'.\data\\7_23_nathans_pool\FSL-01F_Fred\\P7130377.ORF')              
json_path_1 = os.path.join(os.path.dirname(__file__), r'.\\camera_imaging_pipeline\\params1.json')
params1 = json.load(open(json_path_1))

#create a processing pipeline based on the parameters dictionary in params1, apply that pipeline to an image
#and finally display that image. 
config1 = imageProcessing(params1)
img1_data, img1_visual = config1.applyToImage(img1_path)
config1.showImage(img1_visual)

z = img1_data
nrows, ncols = z.shape
x = np.linspace(0, nrows, nrows)
y = np.linspace(0, ncols, ncols)
x, y = np.meshgrid(x, y)

region = np.s_[420:500, 250:320]
x, y, z = x[region], y[region], z[region]

# Set up plot
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

ls = LightSource(270, 45)
# To use a custom hillshading mode, override the built-in shading and pass
# in the rgb colors of the shaded surface calculated from "shade".
rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
                       linewidth=0, antialiased=False, shade=False)

plt.show()

#config1.showImage(img2_visual)
#config1.showImage(img3_visual)