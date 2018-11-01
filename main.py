import rawpy
import imageio
from utils import ensure_dir, file_message, files
import cv2
import numpy as np
from PIL import Image
import shutil
import json
import os
import pickle

DIR = "./photos/plant1_20180615/entire_plant"
APP_DIR = "C:/Users/Breght/Documents/Doctoraat/Annotator"

RAW_DIR = DIR + "/raw"
IMG_DIR = DIR + "/images"
ANN_DIR = DIR + "/annotations"
CAL_DIR = DIR + "/calibration"
UND_DIR = DIR + "/undistorted"
UNDCR_DIR = DIR + "/undistorted_cropped"
ARUCO_DIR = DIR + "/ArUco"
RESIZE_DIR = DIR + "/resized"
IMG_TRANSFER_DIR = DIR + "/img_transfer"
ANN_TRANSFER_DIR = DIR + "/ann_transfer"


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
    else:
        cvt = img.copy()

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


def batch_threshold_segmentation(limits=([0, 255], [54, 255], [0, 255]), colour_space="HSV", in_dir=IMG_DIR, out_dir=ANN_DIR):
    ensure_dir(out_dir)

    for in_f, out_f in files(in_dir=in_dir, out_dir=out_dir, in_ext=".jpg", out_ext=".png"):
        threshold_segmentation(in_f, out_f, limits, colour_space)


def undistortion_parameters(cal_f):
    dims = (7, 9)
    coords_3d = np.zeros((dims[0]*dims[1], 3), np.float32)
    coords_3d[:, :2] = np.mgrid[0:dims[0], 0:dims[1]].T.reshape(-1, 2)

    squares_3d = []
    squares_2d = []
    img = cv2.imread(cal_f, 0)
    ret, img = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)
    ret, corners = cv2.findChessboardCorners(img, dims, None)
    if ret:
        squares_3d.append(coords_3d)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
        squares_2d.append(corners)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(squares_3d, squares_2d, img.shape[::-1], None, None)
        h, w = img.shape[:2]
        newmtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newmtx, (w, h), 5)
        return mapx, mapy, roi
    else:
        raise Exception("Chessboard detection failed")


def undistortion(in_f, out_f, mapx, mapy, roi, crop=True):
    img = cv2.imread(in_f, 1)
    undist = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    if crop:
        x, y, w, h = roi
        undist = undist[y:y+h, x:x+w]
    cv2.imwrite(out_f, undist)
    file_message(out_f)


def batch_undistortion(cal_dir=CAL_DIR, in_dir=IMG_DIR, out_dir=UND_DIR, crop=True):
    ensure_dir(out_dir)

    cal_f = files(dir=cal_dir, ext=".jpg")[0]
    mapx, mapy, roi = undistortion_parameters(cal_f)
    for in_f, out_f in files(in_dir=in_dir, out_dir=out_dir, in_ext=".jpg", out_ext=".jpg"):
        undistortion(in_f, out_f, mapx, mapy, roi, crop)


def detect_markers(in_f, out_f, aruco_dict, aruco_params, show=False):
    img = cv2.imread(in_f, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    with open(out_f, "wb") as out:
        pickle.dump([corners, ids, rejected], out)
    file_message(out_f)

    if show:
        img_with_aruco = cv2.aruco.drawDetectedMarkers(img, corners, ids, (0, 255, 0))
        cv2.imshow("aruco", img_with_aruco)
        cv2.waitKey(0)


def batch_detect_markers(in_dir=UNDCR_DIR, out_dir=ARUCO_DIR, show=True):
    ensure_dir(out_dir)

    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_1000)
    aruco_params = cv2.aruco.DetectorParameters_create()
    for in_f, out_f in files(in_dir=in_dir, out_dir=out_dir, in_ext=".jpg", out_ext=".pkl"):
        detect_markers(in_f, out_f, aruco_dict=aruco_dict, aruco_params=aruco_params, show=show)


def resize(in_f, out_f, ratio=.5):
    img = cv2.imread(in_f, 1)
    dim = (int(ratio * img.shape[1]), int(ratio * img.shape[0]))
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    cv2.imwrite(out_f, resized)
    file_message(out_f)


def batch_resize(in_dir, out_dir, ratio=.5):
    ensure_dir(out_dir)

    for in_f, out_f in files(in_dir=in_dir, out_dir=out_dir, in_ext=".jpg", out_ext=".jpg"):
        resize(in_f, out_f, ratio)
    for in_f, out_f in files(in_dir=in_dir, out_dir=out_dir, in_ext=".png", out_ext=".png"):
        resize(in_f, out_f, ratio)


def prepare_annotator(img_dir=IMG_DIR, ann_dir=ANN_DIR, app_dir=APP_DIR, classes=("background", "plant", "panicle")):
    img_out_dir = app_dir + "/data/images"
    for f in files(img_out_dir, ".jpg"):
        os.remove(f)
    for f in files(img_dir, ".jpg"):
        shutil.copy(f, img_out_dir)
        file_message(img_out_dir + "/" + os.path.basename(f))

    ann_out_dir = app_dir + "/data/annotations"
    for f in files(ann_out_dir, ".png"):
        os.remove(f)
    for f in files(ann_dir, ".png"):
        shutil.copy(f, ann_out_dir)
        file_message(ann_out_dir + "/" + os.path.basename(f))

    jsonfile = files(app_dir + "/data", ".json")[0]
    with open(jsonfile, "r") as read_file:
        data = json.load(read_file)
    data["labels"] = classes
    data["imageURLs"] = ["data/images/" + os.path.basename(f) for f in files(img_out_dir, ".jpg")]
    data["annotationURLs"] = ["data/annotations/" + os.path.basename(f) for f in files(ann_out_dir, ".png")]
    with open(jsonfile, "w") as write_file:
        json.dump(data, write_file)
    file_message(jsonfile)


if __name__ == "__main__":
    # step 1: convert images from raw to jpeg
    batch_raw2jpg(RAW_DIR, IMG_DIR)
    batch_raw2jpg(CAL_DIR, CAL_DIR)
    # step 2: make images smaller for more convenient use in annotator tool
    batch_resize(IMG_DIR, RESIZE_DIR, .5)
    # step 3: do the segmentation of the resized images
    batch_threshold_segmentation(limits=[[0, 255], [54, 255], [0, 255]], colour_space="HSV", in_dir=RESIZE_DIR, out_dir=ANN_DIR)
    # step 4: prepare the annotator tool (give the tool the images and annotations)
    #   first create the IMG_TRANSFER_DIR and ANN_TRANSFER_DIR directories,
    #   and move some images and corresponding annotations in the folder,
    #   then run the prepare_annotator function
    prepare_annotator(IMG_TRANSFER_DIR, ANN_TRANSFER_DIR, APP_DIR, ("background", "plant", "panicle"))

    # step 5: do undistortion
    batch_undistortion(CAL_DIR, IMG_DIR, UND_DIR, crop=False)
    batch_undistortion(CAL_DIR, IMG_DIR, UNDCR_DIR, crop=True)
    # step 6: detect markers
    batch_detect_markers(UNDCR_DIR, ARUCO_DIR)