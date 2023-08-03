import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit, polyval

from .analysis import display_data
from .helpers import clamp, scale_data


#resize the image for convenient verification of the processed image by the user.
def imageResize(img, resize_val):
    scale_percent = resize_val # percent of original size
    width = int(img.shape[1] * (scale_percent / 100))
    height = int(img.shape[0] * (scale_percent / 100))
    dim = (width, height)   

    img = cv.resize(img, dim, interpolation = cv.INTER_AREA)
    return img

#linearize the raw sensor date
def linearization(img):
    img = ((img - img.min()) * (1/(img.max() - img.min())))
    img = scale_data(img, 16)
    return img

#apply a demosaicing algorithm to create a 3 color channel RGB image 
def demosaic(img):
    img = cv.demosaicing(img, cv.COLOR_BayerGB2BGR) 
    return img

#denoise the image by convolving the data with a low-pass filter
def denoising(img, val):
    img = cv.blur(img, (val,val))
    return img

#provides the possibility to convert the image to greyscale
def colorSpace(img, colour):
    if colour == False:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img

#apply exposure compensation to the image, and handle any clipping that might be ocurring
def exposureComp(img, val):
    img = img * val
    _, img = cv.threshold(img, 65535, 65535, cv.THRESH_TRUNC)
    img = img.astype('uint16')

    return img

#provides the ability to adjust the low, mid and high tones separately
def toneCurve(img):
    table = [None] * 256
    for i in range(256):
        table[i] = 30

    pts = np.linspace(0 ,np.pi * 2, 255)
    sin_pts = np.sin(pts)
    sin_pts += 1
    
    x  = np.linspace(0,255,255).astype('uint8')
    y = x * sin_pts

    plt.plot(x, y)
    plt.show()

    table = np.asarray(table).astype('uint8')
    img = np.asarray(img).astype('uint8')
    img = cv.LUT(img, table)
    
    return img

#maps the channel value to new values according to the gamma function
def gammaCorrection (img, gamma):
    img_buf = (img)/img.max()
    buf = np.power(img_buf, gamma) * img.max()#65535
    #buf = buf.astype(np.uint16)
    return buf

#provides white balance adjustments to the image based on the grey world algorithm
def greyWorldWB(img, colour):
    if colour == True: 
        b, g, r = cv.split(img)
        r_avg = cv.mean(r)[0]
        g_avg = cv.mean(g)[0]
        b_avg = cv.mean(b)[0]

        kr = g_avg / r_avg
        kg = 1
        kb = g_avg / b_avg  

        r = cv.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
        g = cv.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
        b = cv.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)

        balance_img = cv.merge([b, g, r])

    else:
        balance_img = img   
    return balance_img

# def scale_pixels(img):
#     b, g , r = cv.split(img)
#     b = np.subtract(b, min(b.flatten(order='C')))
#     b = 255*(b/max(b.flatten(order='C'))).astype('uint8')

#     g = np.subtract(g, np.full_like(g, min(g.flatten(order='C'))))
#     g = 255*(g/max(g.flatten(order='C'))).astype('uint8')

#     r = np.subtract(r, np.full_like(r, min(r.flatten(order='C'))))
#     r = 255*(r/max(r.flatten(order='C'))).astype('uint8')

#     img = cv.merge([b, g, r])
#     print(img.min())
#     print(img.max())
#     return img
