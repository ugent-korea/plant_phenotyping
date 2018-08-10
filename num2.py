'''
import rawpy
import imageio


def raw2jpg(in_f, out_f):
    with rawpy.imread(in_f) as raw:
        rgb = raw.postprocess()

    imageio.imsave(out_f, rgb)
'''
''''
import glob

os.mkdir("C:/Users/USER/PycharmProjects/test/photos/plant1_20180615/entire_plant/images")
nef_list = glob.glob("C:/Users/USER/PycharmProjects/test/photos/plant1_20180615/entire_plant/raw/*.NEF")
for a in nef_list:
    print(a)
    outf = "C:/Users/USER/PycharmProjects/test/photos/plant1_20180615/entire_plant/images/"+a[-12:-3] +"jpg"
    print(outf)
    raw2jpg(a, outf)
'''
import cv2
import numpy as np
import numpy as np
from PIL import Image
im = cv2.imread("C:/Users/USER/PycharmProjects/test/photos/plant1_20180615/entire_plant/images/DSC_0210.jpg")

#convert RGB to HSV
hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
# Threshold the HSV image to get only green~yellow colors
mask1 = cv2.inRange(hsv, (36, 0, 0), (70, 255, 255))
mask2 = cv2.inRange(hsv, (20, 100, 100), (30, 255, 255))
mask = mask1 + mask2
annotation = mask

# Encode
Image.fromarray(np.stack([
    np.bitwise_and(annotation, 255),
    np.bitwise_and(annotation >> 8, 255),
    np.bitwise_and(annotation >> 16, 255),
    ], axis=2).astype(np.uint8)).save(
        './photos/plant1_20180615/entire_plant/annotations/DSC_0210.png')
# Decode
encoded = np.array(Image.open('./photos/plant1_20180615/entire_plant/annotations/DSC_0210.png'))
annotation = np.bitwise_or(np.bitwise_or(
    encoded[:, :, 0].astype(np.uint32),
    encoded[:, :, 1].astype(np.uint32) << 8),
    encoded[:, :, 2].astype(np.uint32) << 16)
print(np.unique(annotation))