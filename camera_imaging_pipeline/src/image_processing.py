import numpy as np
import rawpy
import matplotlib.pyplot as plt
import cv2 as cv
import os


class imageProcessing():

    def __init__(self, path, colour):
        self.raw = rawpy.imread(path)
        self.img = [self.raw.raw_image.copy()]
        self.colour = colour

    def getImages(self):
        return self.img
    
    def getFirstImage(self):
        return self.img[0]

    def getLastImage(self):
        return self.img[-1]
    
    def getRGB(self, x, y):
        buf = self.img[-1]
        r = buf[x, y, 0]
        g = buf[x, y, 1]
        b = buf[x, y, 2]

        print('R value: '+str(r))
        print('G value: '+str(g))
        print('B value: '+str(b))

    def showImage(self, img):
        cv.imshow("urer", img)
        k = cv.waitKey(0)
        cv.destroyAllWindows()
    
    def imageResize(self):
        scale_percent = 20 # percent of original size
        width = int(self.img[-1].shape[1] * (scale_percent / 100))
        height = int(self.img[-1].shape[0] * (scale_percent / 100))
        dim = (width, height)   

        resized = cv.resize(self.img[-1], dim, interpolation = cv.INTER_AREA)

        self.img.append(resized)

        return self.img[-1]

    def linearization(self):
        buf = ((self.img[-1] - self.img[-1].min()) * (1/(self.img[-1].max() - self.img[-1].min()) * 255)).astype('uint8')

        self.img.append(buf)
        return self.img[-1]

    def demosaic(self):
        buf = cv.demosaicing(self.img[-1], cv.COLOR_BayerGB2BGR) 
        self.img.append(buf)
        return self.img[-1]

    def lens_correction(self):
        h, w = self.img[-1].shape[:2]

        absolute_path = os.path.dirname(__file__)
        
        mtx = np.load(os.path.join(absolute_path,'..\\files\\camera_matrix.npy'))
        dist = np.load(os.path.join(absolute_path,'..\\files\\distortion_coefficient.npy'))
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        
        # undistort
        dst = cv.undistort(self.img[-1], mtx, dist, None, newcameramtx)
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]

        return dst

    def denoising(self, val):
        buf = cv.fastNlMeansDenoisingColored(self.img[-1], val, 10, 7, 21)
        self.img.append(buf)
        return self.img[-1]
    
    def colorSpace(self):
        #buf = cv.normalize(self.img[-1], None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
        #buf = cv.convertTo(self.img[-1], cv.CV_8U, 1.0/255)
        #buf = cv.equalizeHist(buf)
        buf = self.img[-1].astype(np.uint8)

        if self.colour == True:
            buf = cv.cvtColor(buf, cv.COLOR_BGR2RGB)
        else:
            buf = cv.cvtColor(buf, cv.COLOR_BGR2GRAY)
       
        self.img.append(buf)
        return self.img[-1]
    
    def greyWorldWB(self):              #not in use
        current_image = self.img[-1]

        b, g, r = cv.split(current_image)
        r_avg = cv.mean(r)[0]
        g_avg = cv.mean(g)[0]
        b_avg = cv.mean(b)[0]

        #k = (r_avg + g_avg + b_avg) / 3 
        kr = g_avg / r_avg
        kg = 1
        kb = g_avg / b_avg  

        r = cv.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
        g = cv.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
        b = cv.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)

        balance_img = cv.merge([b, g, r])
        balance_img = cv.cvtColor(balance_img, cv.COLOR_BGR2RGB)

        self.img.append(balance_img)
        
        
        return self.img[-1]
    
    def whiteBalance(self, val):
        if self.colour == True:
            current_image = self.img[-1]

            r, g, b = cv.split(current_image)
            kb = 255/(27*val)
            kg = 255/(46*val)
            kr = 255/(40*val)

            r = cv.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
            g = cv.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
            b = cv.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)

            balance_img = cv.merge([r, g, b])
            balance_img = cv.cvtColor(balance_img, cv.COLOR_BGR2RGB)

            self.img.append(balance_img)

        return self.img[-1]

    def exposureComp(self, val):
        buf = self.img[-1]
        buf += val
        buf = cv.normalize(buf, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
        
        self.img.append(buf)
        return self.img[-1]

    def gammaCorrection (self, gamma):
        img_buf = (self.img[-1])/255
        buf = np.power(img_buf, gamma) * 255
        buf = buf.astype(np.uint8)

        self.img.append(buf)
        return self.img[-1]
    
    def colourMap(self):
        buf = cv.applyColorMap(self.img[-1], cv.COLORMAP_PARULA)

        self.img.append(buf)
        return self.img[-1]
    
    def toneCurve(self, low, mid, high):
        buf = self.img[-1]
        buf = buf.astype(np.uint8)

        buf[buf <= 85] += np.asarray(low).astype(np.uint8)
        buf[(85 < buf) & (buf < 170)] += np.asarray(mid).astype(np.uint8)
        buf[buf >= 170] = np.asarray(high).astype(np.uint8)

        self.img.append(buf)
        return self.img[-1]