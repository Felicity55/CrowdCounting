from torchvision.io import read_image
import numpy as np
from PIL import Image
import cv2
from glob import glob
from tqdm import tqdm
from skimage.measure import label, regionprops, find_contours


""" Convert a mask to border image """
def mask_to_border(mask,image):
    h, w = mask.shape
    border = np.zeros((h, w))
    # print(mask>0)
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(contours)
    contours = contours[0] if len(contours) == 2 else contours[1]
    ROI_number = 0
    copy=image.copy()
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        ROI = image[y:y+h, x:x+w]
        cv2.imwrite(r'C:\Users\CVPR\Desktop\Test images\car9_ROI_{}.png'.format(ROI_number), ROI)
        cv2.rectangle(copy,(x,y),(x+w,y+h),(0,0,255),2)
        ROI_number += 1

    cv2.imshow('thresh', mask)
    cv2.imshow('copy', copy)
    cv2.imwrite(r'C:\Users\CVPR\Desktop\Test images\car9_bbox.png',copy)
    cv2.waitKey(0)


    return 0


def parse_mask(mask):
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask], axis=-1)
    return mask

if __name__ == "__main__":
    img_path = r"C:\Users\CVPR\Desktop\Test images\car9.jpg"
    mask_path = r"C:\Users\CVPR\Desktop\Test images\car9dm.png"


    """ Read image and mask """
    x = cv2.imread(img_path, cv2.IMREAD_COLOR)
    h,w,c=x.shape
    y = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    y=cv2.resize(y,(w,h))
    # print(x.shape, y.shape)
    """ Detecting bounding boxes """
    bboxes = mask_to_border(y,x)

  