import cv2
import numpy as np

## Read
img = cv2.imread('./photos/plant1_20180615/entire_plant/images/DSC_0210.jpg')

## convert to hsv
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

## mask of green (36,0,0) ~ (70, 255,255)
mask1 = cv2.inRange(hsv, (36, 0, 0), (70, 255,255))

## mask of yellow (20,100,100) ~ (30, 255, 255)
mask2 = cv2.inRange(hsv, (20, 100, 100), (30, 255,255))

mask = mask1 + mask2

## slice the green
imask = mask>0
green = np.zeros_like(img, np.uint8)
green[imask] = img[imask]



## save
cv2.imwrite("DSC_0210.png", green)