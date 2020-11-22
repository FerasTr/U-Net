import numpy as np
import cv2 as cv
from PIL import Image
import os
import glob

img_path = "NE43_D5_a2_100_c1.tif"

im = Image.open(img_path)
# red = (255, 0, 0)
# white = (255, 255, 255)
# black = (0, 0, 0)
# pixels = np.array(im)
# pixels[pixels != 255] = 0
# Rmask = np.any(pixels != red, axis=-1)
# Wmask = np.any(pixels != white, axis=-1)
# Bmask = np.any(pixels != black, axis=-1)
# pixels[np.logical_and.reduce((Rmask, Wmask, Bmask))] = black
# im = Image.fromarray(pixels)
im.save("NE43_D5_a2_100_c1.png")
