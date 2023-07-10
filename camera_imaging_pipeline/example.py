from src.image_processing import imageProcessing
import cv2 as cv
import os


path = os.path.join(os.path.dirname(__file__), '.\data\Fishsense.dng')
img1 = imageProcessing(path, True)


img1.linearization()
img1.lens_correction()
img1.demosaic()
#img1.denoising(5)
img1.colorSpace()
img1.exposureComp(0)
img1.toneCurve(-5, 5, -5)
img1.gammaCorrection(1.3)
img1.whiteBalance(1.8)
#img1.greyWorldWB()


img1.imageResize()
img1.showImage(img1.getLastImage())

#write_path = os.path.join(os.path.dirname(__file__), '.\\files\\output_image.jpg')
#cv.imwrite(write_path, img1.getLastImage())