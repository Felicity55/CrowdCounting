# import cv2
# from matplotlib import pyplot as plt
# import imutils
# mser = cv2.MSER_create()
# img_path = r'C:\Users\CVPR\Desktop\Test images\realdata\crop\lp\027_ROI_1_crop_2.png'
 
#     # read/load an image
# image = cv2.imread(img_path)

# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         # gray = cv2.bilateralFilter(gray, 13, 15, 15)
# edged = cv2.Canny(gray, 30, 200) #Perform Edge detection
# contours=cv2.findContours(edged.copy(),cv2.RETR_TREE,
#                                             cv2.CHAIN_APPROX_SIMPLE)
# contours = imutils.grab_contours(contours)
# contours = sorted(contours,key=cv2.contourArea, reverse = True)[:10]
# regions = mser.detectRegions(gray)
# screenCnt = 0
# for c in contours:
#     # approximate the contour
#     peri = cv2.arcLength(c, True)
#     approx = cv2.approxPolyDP(c, 0.018 * peri, True)
#     # if our approximated contour has four points, then
#     # we can assume that we have found our screen
#     if len(approx) == 4:
#         screenCnt = approx
#         break

# # hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions[0]]
# # for hull in hulls:
# #             # approximate the contour
# #             peri = cv2.arcLength(hull, True)
# #             approx = cv2.approxPolyDP(hull, 0.018 * peri, True)
# #             # if our approximated contour has four points, then
# #             # we can assume that we have found our screen
# #             if len(approx) == 4:
# #                 screenCnt = approx
# #                 break
#                 # screenCnt.append(approx)
# cv2.drawContours(image,[screenCnt],0,(0,255,0),2)
# cv2.imshow('copy', image)
# plt.imsave(r'C:\Users\CVPR\Desktop\Test images\realdata\crop\lp\027_ROI_1_cropbbox_2.png', image)    
# cv2.waitKey(0)


import argparse

import matplotlib.pyplot as plt

from lib.text_detection import TextDetection
from utils import plt_show
from config import Configure


## Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, required=True, help="Path to the input image")
parser.add_argument("-o", "--output", type=str, help="Path to the output image")
parser.add_argument("-d", "--direction", default='both+', type=str, choices=set(("light", "dark", "both", "both+")), help="Text searching")
parser.add_argument("-t", "--tesseract", action='store_true', help="Tesseract assistance")
parser.add_argument("--details", action='store_true', help="Detailed run with intermediate steps")
parser.add_argument("-f", "--fulltesseract", action='store_true', help="Full Tesseract")
args = vars(parser.parse_args())
IMAGE_FILE = r'C:\Users\CVPR\Desktop\Test images\realdata\crop\lp\004_ROI_2_crop_0'
#args["input"]
OUTPUT_FILE = args["output"]
DIRECTION = args["direction"]
TESS = args["tesseract"]
DETAILS = args["details"]
FULL_OCR = args["fulltesseract"]

if __name__ == "__main__":
    config = Configure()
    td = TextDetection(IMAGE_FILE, config, direction=DIRECTION, use_tesseract=TESS, details=DETAILS)
    if FULL_OCR:
        bounded, res = td.full_OCR()
        plt_show((td.img, "Original"), (bounded, "Final"), (res, "Mask"))
    else:
        res = td.detect()
        plt_show((td.img, "Original"), (td.final, "Final"), (res, "Mask"))
        if OUTPUT_FILE:
            plt.imsave(OUTPUT_FILE, td.final)
            print("{} saved".format(OUTPUT_FILE))