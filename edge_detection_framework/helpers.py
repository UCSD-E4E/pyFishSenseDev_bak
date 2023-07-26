import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LightSource


def scale_data(img, type):
    if type == 8:
        scale = 255
    elif type == 16:
        scale = 65535
    elif type == 1:
        scale = 1
    
    img = np.subtract(img, np.min(img))
    img = scale*(img/np.max(img))

    if type == 8:
        img = img.astype('uint8')
    elif type == 16:
        img = img.astype('uint16')
    elif type == 1:
        img = img.astype('float16')

    return img

def cropImage(img, roi):
    x, y, w, h = roi
    img_new = img[y:y+h,x:x+w]
    
    return img_new


def plot3D(arr):
    z = arr
    nrows, ncols = z.shape
    x = np.linspace(0, ncols, ncols)
    y = np.linspace(0, nrows, nrows)
    x, y = np.meshgrid(x, y)

    ##region = np.s_[5:50, 5:50]
    #x, y, z = x[region], y[region], z[region]

    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ls = LightSource(270, 45)

    rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
                       linewidth=0, antialiased=False, shade=False)

    plt.show()

def plotCut(arr, rad):
    cut = arr[int(np.floor(arr.shape[0]/2)), :]
    x = np.linspace(0, len(cut), len(cut))
    
    label = "Filter Radius of :"+str(rad)

    plt.plot(x, cut)
    plt.title(label)
    plt.show()