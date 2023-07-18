import numpy as np


def clamp(n, floor, ceiling):
  if n < floor: return floor
  if n > ceiling: return ceiling
  return n

def scale_data(img, type):
    if type == 8:
        scale = 255
    elif type == 16:
        scale = 65535
    
    img = np.subtract(img, min(img.flatten(order='C')))
    img = scale*(img/max(img.flatten(order='C')))

    if type == 8:
        img = img.astype('uint8')
    elif type == 16:
        img = img.astype('uint16')

    return img