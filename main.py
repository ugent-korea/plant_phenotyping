import rawpy
import imageio
from plant_phenotyping.utils import ensure_dir, file_message, files
import cv2
import numpy as np
from PIL import Image

DIR = "I:/Doctoraat/Plant photos/20180615 Plant 1"
APP_DIR = "C:/Users/Breght/Documents/Doctoraat/Annotator"

RAW_DIR = DIR + "/raw"
IMG_DIR = DIR + "/images"
ANN_DIR = DIR + "/annotations"
CAL_DIR = DIR + "/calibration"
UND_DIR = DIR + "/undistorted"
UNDCR_DIR = DIR + "/undistorted_cropped"
ARUCO_DIR = DIR + "/ArUco"


def raw2jpg(in_f, out_f):
    with rawpy.imread(in_f) as raw:
        rgb = raw.postprocess()
    imageio.imsave(out_f, rgb)
    file_message(out_f)


def batch_raw2jpg(in_dir=RAW_DIR, out_dir=IMG_DIR):
    ensure_dir(out_dir)

    for in_f, out_f in files(in_dir=in_dir, out_dir=out_dir, in_ext=".nef", out_ext=".jpg"):
        raw2jpg(in_f, out_f)


def threshold_segmentation(in_f, out_f, limits, colour_space="HSV"):
    img = cv2.imread(in_f, 1)
    if colour_space == "BGR":
        cvt = img.copy()
    elif colour_space == "LAB":
        cvt = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    elif colour_space == "HSV":
        cvt = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    limits = np.array(list(zip(*limits)))
    mask = cv2.inRange(cvt, limits[0], limits[1])
    mask = (mask / 255).astype(int)

    encode_segmentation(mask, out_f)
    file_message(out_f)


def encode_segmentation(segm, filename):
    Image.fromarray(np.stack([
        np.bitwise_and(segm, 255),
        np.bitwise_and(segm >> 8, 255),
        np.bitwise_and(segm >> 16, 255),
    ], axis=2).astype(np.uint8)).save(filename)


def decode_segmentation(filename):
    encoded = np.array(Image.open(filename))
    annotation = np.bitwise_or(np.bitwise_or(
        encoded[:, :, 0].astype(np.uint32),
        encoded[:, :, 1].astype(np.uint32) << 8),
        encoded[:, :, 2].astype(np.uint32) << 16)

    return annotation


def batch_threshold_segmentation(limits=[[0, 255], [54, 255], [0, 255]], colour_space="HSV", in_dir=IMG_DIR, out_dir=ANN_DIR):
    ensure_dir(out_dir)

    for in_f, out_f in files(in_dir=in_dir, out_dir=out_dir, in_ext=".jpg", out_ext=".png"):
        threshold_segmentation(in_f, out_f, limits, colour_space)


if __name__ == "__main__":
    batch_raw2jpg(RAW_DIR, IMG_DIR)
    batch_raw2jpg(CAL_DIR, CAL_DIR)
    batch_threshold_segmentation(limits=[[0, 255], [54, 255], [0, 255]], colour_space="HSV", in_dir=IMG_DIR, out_dir=ANN_DIR)
