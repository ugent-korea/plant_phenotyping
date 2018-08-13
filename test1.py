import numpy as np
import cv2
import glob
import imageio

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
a = 6
b = 7
objp = np.zeros((b*a,3), np.float32)
objp[:,:2] = np.mgrid[0:a,0:b].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob("./photos/plant1_20180615/entire_plant/annotations/*.png")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (a,b), None)
    print(fname, ret)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners)

    # Draw and display the corners
    cv2.drawChessboardCorners(img, (a,b), corners, ret)
    cv2.imshow('img',img)
    #imageio.imsave('./photos/plant1_20180615/entire_plant/annotations/image.jpg', img)
    cv2.waitKey(-1)

cv2.destroyAllWindows()