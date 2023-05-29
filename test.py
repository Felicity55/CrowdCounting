from cv2 import countNonZero
import matplotlib.pyplot as plt
import matplotlib.cm as CM
# from models.CSRNet import CSRNet
from model import CSRNet
from model import CSRNet_Sig
from network import U_Net
from torch.utils.data import DataLoader
from dataloader import Cars
# from counting_datasets.CityCam_maker import  
from dataloader import ToTensor
# from my_dataloader import CrowdDataset
from sklearn.cluster import KMeans
import imutils
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as CM
from torchvision import transforms
from torch.utils.data import DataLoader
import sys
import os
import cv2
import imutils
sys.path.append('..')
from sklearn.cluster import KMeans
from PIL import Image
import scipy.spatial
import scipy.ndimage
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
import skimage
from easyocr import Reader
from scipy.ndimage import gaussian_filter
torch.cuda.empty_cache()

# Global variables init
use_gpu = torch.cuda.is_available()
# checkpoint_path = r'C:\Users\CVPR\Soumi DI\License plate detection\checkpoint\LP-NightCSRNet-w015-Epochs-5000_BatchSize-8_LR-0.0001_Momentum-0.95_Gamma-0.5_Version-1\best_model.pt'
checkpoint_path = r'C:\Users\CVPR\Soumi DI\License plate detection\checkpoint\NewdataCSRNet-w015-Epochs-7000_BatchSize-8_LR-0.0001_Momentum-0.95_Gamma-0.5_Version-1\best_model.pt'
device = torch.device('cuda:0' if use_gpu else 'cpu')


def infer(sample):
    # model = U_Net()
    model=CSRNet_Sig()
    # checkpoint = torch.load(checkpoint_path) #28.9.22
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda:0' if use_gpu else 'cpu')) #28.9.22// run with only cpu
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    img = sample
    # print("Image shape:",img.shape)
    # density_map = dm
    # gt_count = sample['gt_count']
    
    # make img's dimension (1, C, H, W)
    img = torch.unsqueeze(img, dim=0)
    

    # make the dimension of the density map (1, H, W)
    # density_map = torch.unsqueeze(density_map, dim=0)
   
    
    img = img.to(device)
    # density_map = density_map.to(device)
    # print("image is:", img.data>0)
    print("Shape of test image", img.shape)
    
    with torch.set_grad_enabled(False):
        et_dm = model(img).detach()
        print('etdm shape', et_dm.shape)
        # down_sample = nn.Sequential(nn.MaxUnpool2d(2), nn.MaxUnpool2d(2))
        # et_dm = down_sample(et_dm)
        
        et_dm = et_dm.squeeze(0).cpu().numpy()
        print('etdm shape', et_dm.shape)
        # print("Estimated density map shape is:",et_dm.shape)
        # print("Estimated density map is:",et_dm)
        # print("Max value of Estimated Density Map is:",et_dm.max())
        # print("Min value of Estimated Density Map is:",et_dm.min())
        # down_gt_dm = down_gt_dm.cpu().numpy()
        # et_dm_reshape = et_dm.copy().reshape(-1, 1)
    # print(f'The integral of the estimated density map: {et_dm.sum()}')
    
    # print(et_dm.dtype)
   
    x=et_dm.squeeze()
    x = gaussian_filter(x, sigma=20)
    
    # plt.imshow(x,cmap=CM.jet_r)
    # plt.pause(0.001)
    # plt.show()

    x1d = x.copy().reshape(-1, 1)
    # print("Values:",x1d)

    #K-Means Clustering
    kmeans = KMeans(n_clusters=2, random_state=0, max_iter=2).fit(x1d)
    y_pred=kmeans.predict(x1d)

    labels=kmeans.labels_
    centroids = kmeans.cluster_centers_
    # print("Kmeans labels:",kmeans.labels_)
    # zeros=labels.count(0)
    num_zeros = (labels == 0)
    num_ones = (labels > 0)
    # print(num_zeros.sum(),num_ones.sum())
    center= np.array(kmeans.cluster_centers_)
    
    zlevel_mean=np.mean(num_zeros)
    nzlevel_mean=np.mean(num_ones)
    # print("Mean values:", zlevel_mean, nzlevel_mean)
    zlevel_std=np.std(num_zeros)
    nzlevel_std=np.std(num_ones)
    # print("Standard Deviations:", zlevel_std, nzlevel_std)
    # print("centers:", center)
    # img=img.squeeze()
    width, height = x.shape
    y_pred = y_pred.reshape(width, height)
# print('y_pred shape ', y_pred.shape)
# print("y prediction",y_pred)
    a2 = centroids[labels]
# print('a2 shape ', a2.shape)
    a3 = a2.reshape(width, height)
    bin_img = np.where(a3== np.min(a3), 255, 0)
    # num_zero = (bin_img == 0).sum() #black
    # num_one = (bin_img != 0).sum() #white
    # print(num_zero,num_one)
    bin_img=1-bin_img
    # print('a3 shape ', a3.shape)
    # print('a3 values ', a3)
    dmimage=bin_img*255
    # dmimage=Image.fromarray(bin_img*255)
    # dmimage=dmimage.resize((w,h))
    plt.imshow(dmimage,cmap=CM.gray)
    # plt.show()

   
    return x
def auto_canny(image, sigma=0.5):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

def mask_to_box(mask,image):
    
    h,w= mask.shape
    pred=np.zeros((h, w))
    
    # mask=Image.fromarray(mask,"L")
    largest=0
    hull = []
    border = np.zeros((h, w))
    print(type(mask))
    contours= cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    ROI_number = 0
    copy=image.copy()
    gray = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)
    mser = cv2.MSER_create()
    for cn in contours:
        # a=cv2.contourArea(cn)
        # if a>largest:
        #     largest=a
        #     largest_cont=cn
        # hull.append(cv2.convexHull(largest_cont, False))
        x,y,w,h = cv2.boundingRect(cn)
        ROI = image[y:y+h, x:x+w]
        # print("ROI:", ROI)
        # plt.imsave(r'C:\Users\CVPR\Desktop\Test images\realdata\crop\lp\053_ROI_0_lpcrop_{}.png'.format(ROI_number), ROI)
        ############# OCR detection  ####################################################################################################################
        reader = Reader(['en'])
        result = reader.readtext(ROI)
        # result = np.asarray(result, dtype = 'int')
        print(result)
        if result != []:
            print('Detected')
            i=100
            for detection in result:
                print(detection[0])
                t1=np.asarray(detection[0])
                l=t1.astype(int)
                print(type(t1))
                
                a=[]
                for i in l:
                    i=i+[x, y]
                    a.append((i[0],i[1]))
                print("print a",a)
                a=np.asarray(a)
                img2= cv2.polylines(copy, [a], 1, (0,255,0),5)
                pred[a[0][1]:a[2][1],a[0][0]:a[1][0]] = 1.
                i=i+10
            plt.imshow(img2)
            plt.show()
            plt.imshow(pred)
            plt.show()
            plt.imsave(r'C:\Users\CVPR\Desktop\demo\ImageSets\Main\5_o.png', img2)
            # plt.imsave(r'C:\Users\CVPR\Desktop\Test images\realdata\crop\lp\053_ROI_0_lpbbox_{}.png'.format(ROI_number), img2)
        else:
            print('Not Detected')
        ROI_number += 1
        # else:
        #     print('Not Detected')





    #     gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
    #     # gray = cv2.bilateralFilter(gray, 13, 15, 15)
    #     _, bw = cv2.threshold(gray, 0.0, 255.0, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    #     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    #     connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
        
    #     edged = cv2.Canny(gray, 10, 200) #Perform Edge detection
    #     plt.imshow(edged)
    #     plt.show()
    #     cnts = cv2.findContours(connected.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #     cnts= imutils.grab_contours(cnts)
    #     cnts= sorted(cnts, key = cv2.contourArea, reverse = True)[:]
    #     # print(cnts)
    #     # cv2.imwrite(r'C:\Users\CVPR\Desktop\Test images\realdata\crop\lp\005_ROI_0_crop_{}.png'.format(ROI_number), ROI)
    #     # plt.imsave(r'C:\Users\CVPR\Desktop\Test images\realdata\crop\lp\040_ROI_5_crop_{}.png'.format(ROI_number), ROI)
    #     cv2.rectangle(copy,(x,y),(x+w,y+h),(0,0,255),2)
    #     regions = mser.detectRegions(edged)
    #     # print(regions)
    #     # for p in regions[0]:
    #     # # hull.append(cv2.convexHull(c, False))
    #     #     hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) ]
    #     #     for hull in hulls:
                
    #     #         a=cv2.contourArea(hull)
    #     #         if a>largest:
    #     #             largest=a
    #     #             largest_cont=hull
    #     #             x1,y1,w1,h1 = cv2.boundingRect(largest_cont)
    #     #             lpROI = image[y1:y+h1, x1:x+w1]
        
    #     #     cv2.rectangle(copy,(x,y),(x+w1,y+h1),(0,255,255),1)
    #     screenCnt = 0
    # #     # loop over our contours
    #     # for c in cnts:
    #     #      # approximate the contour
    #     #     peri = cv2.arcLength(c, True)
    #     #     approx = cv2.approxPolyDP(c, 0.018 * peri, True)
    #     #     print(len(approx))
    #     #     # if our approximated contour has four points, then
    #     #      # we can assume that we have found our screen
    #     #     if len(approx) == 4:
    #     #           screenCnt.append(approx)
    #     #         #   break
    #     # cv2.drawContours(copy,[np.array([[x,y]])+s for s in screenCnt],0,(0,255,0), 2)
    #     hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions[0]]
    #     for hull in hulls:
    #         # approximate the contour
    #         peri = cv2.arcLength(hull, True)
    #         approx = cv2.approxPolyDP(hull, 0.018 * peri, True)
    #         # if our approximated contour has four points, then
    #         # we can assume that we have found our screen
    #         print(len(approx))
    #         if len(approx) == 4:
    #             screenCnt=approx
    #             # continue
    #             break
    #             # crop=ROI[approx]
    #             # cv2.imshow('crop',crop)
    #     cv2.drawContours(copy,[np.array([[x,y]])+screenCnt],0,(0,50,255),2)
        
        # cv2.polylines(copy, [np.array([[x,y]])+hull for hull in hulls], 1, (0,255,0),2) #
        # ROI_number += 1
    # # print(hulls)
    # cv2.imshow('thresh', mask)
    # cv2.imshow('copy', copy)
    
    # cv2.waitKey(0)
    # plt.imsave(r'C:\Users\CVPR\Desktop\Test images\realdata\crop\040_LP_5.png',copy)
    
    # reader = Reader(['en'])
    # result = reader.readtext(ROI)
    # result = np.asarray(result, dtype = 'int')
    # print(result)
    # spacer = 100
    # for detection in result: 
    #     # top_left = tuple(np.array([x,y])+detection[0][0])
    #     # bottom_right = tuple(np.array([x,y])+detection[0][2])
    #     top_left = np.array(np.array([x,y])+detection[0][0])
    #     top_left=tuple(top_left.astype(int))
    #     bottom_right = np.array(np.array([x,y])+detection[0][2])
    #     bottom_right=tuple(bottom_right.astype(int))
    #     text = detection[1]
    #     # img = cv2.rectangle(copy,top_left,bottom_right,(0,255,0),3)
    #      #
    #     # img = cv2.putText(copy,text,(20,spacer), 0.5,(0,255,0),2,cv2.LINE_AA)
    #     spacer+=10
    # plt.figure(figsize=(10,10))
    # plt.imshow(img)
    # plt.show()
    # lppoints=result[0][0]
    # lppoints=lppoints.astype(int)
    # l=[]
    # for i in lppoints:
    #     k=[]
    #     for j in i:
    #         x=int(j)
    #         k.append(x)
    #     l.append(k)
    # l=[]
    # if result == True:
    #     print('Detected')
    #     t1=np.asarray(result[0][0])
    #     l=t1.astype(int)
    #     print(type(t1))
        
    #     a=[]
    #     for i in l:
    #         i=i+[x, y]
    #         a.append((i[0],i[1]))
    #     # print(a)
    #     a=np.asarray(a)
    #     img2= cv2.polylines(copy, [a], 1, (0,255,0),2)
    #     plt.imshow(img2)
    #     plt.show()
    #     # plt.imsave(r'C:\Users\CVPR\Desktop\Test images\realdata\crop\040_LP_5.png',copy)
    # else:
    #     print('Not Detected')

    return pred
    

# # create hull array for convex hull points

# # calculate points for each contour
#     # for i in range(len(contours)):
#     #     # creating convex hull object for each contour
#     #     hull.append(cv2.convexHull(contours[i], False))
#     rect = cv2.minAreaRect(largest_cont)
#     box = np.int0(cv2.boxPoints(rect))
#     cv2.drawContours(copy, [box], -1, (255,0,0), 2) 
#     cv2.imshow('copy', copy)# OR
#     # plt.imsave(r'D:/Soumi/Test Sample/Bbox/Cars399_LPbbox1.png',copy)
#     # plt.imsave(r'C:\Users\CVPR\Desktop\Test images\car7_0_LPbbox0.png',copy)
#     cv2.drawContours(image, hull, -1, (0,255,0), 2) 
#     cv2.imshow('copy1', image)
#     # plt.imsave(r'D:/Soumi/Test Sample/Bbox/Cars399_LPbbox2.png',image)
#     # plt.imsave(r'C:\Users\CVPR\Desktop\Test images\car0_LPbbox3.png',image)
#     ROI_number += 1
#     # cv2.imshow('thresh', mask)
#     plt.imsave(r'C:\Users\CVPR\Desktop\Test images\car358_LPbbox.png',image)
#     cv2.waitKey(0)

    ##New code 30/11/22
   
       
    
#     c = max(hull, key=cv2.contourArea)
#     print(c.shape)
#             # Obtain outer coordinates
#     left = tuple(c[c[:, :, 0].argmin()][0])
#     right = tuple(c[c[:, :, 0].argmax()][0])
#     top = tuple(c[c[:, :, 1].argmin()][0])
#     bottom = tuple(c[c[:, :, 1].argmax()][0])
#     pts=np.array([left,top,right,bottom])
#         # Draw dots onto image
#     cv2.drawContours(image, [c], -1, (36, 255, 12), 1)
#     cv2.circle(image, left, 8, (0, 0, 255), -1)
#     cv2.circle(image, right, 8, (0, 0, 255), -1)
#     cv2.circle(image, top, 8, (255, 0, 0), -1)
#     cv2.circle(image, bottom, 8, (255, 0, 0), -1)
    
#     polyimg = cv2.polylines(image, [pts], True, (0, 0, 255), 3)
#     cv2.imshow('polygon', polyimg)

#     print('left: {}'.format(left))
#     print('right: {}'.format(right))
#     print('top: {}'.format(top))
#     print('bottom: {}'.format(bottom))
#     cv2.imshow('thresh', mask)
#     # cv2.imshow('image', image)
#     cv2.waitKey()
# ################################################################ POLYGON ########################################################################
#     # # Set the minimum area for a contour
#     min_area = 1000
    
#     # Draw the contours on the original image and the blank image
#     for c in contours:
#         a=cv2.contourArea(c)
#         if a>largest:
#             largest=a
#             largest_cont=c
#         # area = cv2.contourArea(c)
#         # if area > min_area:
#             cv2.drawContours(image,[largest_cont], 0, (36,255,12), 2, cv2.LINE_4)
            
    
#     # Conver the blank image to grayscale for corner detection
#     # gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
#     # Detect corners using the contours
#     corners = cv2.goodFeaturesToTrack(mask,maxCorners=6,qualityLevel=0.20,minDistance=80) # Determines strong corners on an image
#     corners = np.int0(corners)
#     # Draw the corners on the original image
#     points=[]
#     for corner in corners:
#         x,y = corner.ravel()
#         cv2.circle(image,(x,y),10,(255,0,0),-1)
#         points.append([x,y])
#     print(points)
#     start_lst=points
#     listx = [point[0] for point in start_lst]
#     listy = [point[1] for point in start_lst]

#     # plt.plot(listx,listy)
#     # plt.show()
#     start_point = [listx[0], listy[0]]
#     print(start_point)
#     sorted_points = []
#     while len(start_point)>0:
#         sorted_points.append(start_point)
#         x1, y1 = start_point
#         dists = {(x2, y2): np.sqrt((x1-x2)**2 + (y1-y2)**2) for x2, y2 in zip(listx, listy)}
#         dists = sorted(dists.items(), key=lambda item: item[1])
#         for dist in dists:
#             if dist[0] not in sorted_points: 
#                 start_point = dist[0]
#                 break
#             if dist == dists[-1]:
#                 start_point = ()
#                 break

#     xs = [point[0] for point in sorted_points]
#     ys = [point[1] for point in sorted_points]
#     print(sorted_points)
#     # plt.plot(xs,ys)
#     # plt.show()
#     a=[]
#     for u,v in sorted_points:
#         a.append([u,v])
#     print(a)

#     points = np.array(a)
#     points = points.reshape((-1, 1, 2))
    

#     # drawPolyline
#     polyimage = cv2.polylines(image, [points], True, (0, 0, 255), 3)


    
#     # Display the image
#     # image_copy = cv2.imread("tshirt.jpg")
#     # cv2.imshow('original image', image_copy)
#     # cv2.imshow('image with contours and corners', image)
#     # cv2.imshow('blank_image with contours', mask)
#     # cv2.imshow('poly with contours', polyimage)
#     # # Save the image that has the contours and corners
#     # # cv2.imwrite('polyimage.jpg', polyimage)
#     # plt.imsave(r'C:\Users\CVPR\Desktop\Test images\car0_ROI_1_polyimage.png',polyimage)
#     # # Save the image that has just the contours
#     # # cv2.imwrite('contour_tshirt_blank_image.jpg', mask)
#     # cv2.waitKey() 
#     # return 0  
    

    

data_trans = transforms.Compose([transforms.ToTensor()])
imgpath=r'C:\Users\CVPR\Desktop\demo\ImageSets\Main\5.jpg'
# imgpath=r'D:\Soumi\License plate detection\1\1\Cars199.png'
# imgpath=r'C:\Users\CVPR\Desktop\Test images\realdata\crop\053_ROI_0.png'
# imgpath=r'C:\Users\CVPR\Desktop\Test images\realdata\crop\039.jpg'
# imgpath=r'C:\Users\CVPR\source\repos\Detection\CityCam\253\253-20160421-15\000001.jpg'
# dmpath=r"C:\Users\CVPR\source\repos\Detection\CityCam\846\846-20160429-07\000001_dm.npy"
image=cv2.imread(imgpath)
# image=cv2.resize(image, (512,512))
# image=cv2.resize(image, (2080,1560))

image= cv2. cvtColor(image, cv2.COLOR_BGR2RGB)

# print("image shape",image.shape)
# image=cv2.resize(image,(512,256))
# img = cv2.GaussianBlur(img,(5,5),0)
# img=img.astype('uint8')
# print(img.shape)
# print(type(img))
img=data_trans(image)
# h,w=image.shape
# dmap=np.load(dmpath)
# dmap=data_trans(dmap)
dm=infer(img)
print("Output shape:", dm.shape)
print((np.max(dm)+np.min(dm))/2)

# t = (np.max(dm)+np.min(dm))/2
# t=np.average(dm)
# binary_mask = dm > t


plt.imshow(image)
plt.imshow(dm, cmap=CM.jet, alpha=0.6)
plt.show()
# plt.imsave(r'C:\Users\CVPR\Desktop\demo/1_v_pm.png', ROI)
# blank = dm.point(lambda _: 0)
# comp=Image.composite(OriginalImg,blank,pil_image)
# binmask_img=np.array(binary_mask*255).astype('uint8')
# plt.imshow(image)
# plt.imshow(binmask_img, alpha=0.6)
# # plt.imshow(dm, cmap=CM.jet, alpha=0.6)
# plt.show()


###### Kmeans clustering #########
x1d = dm.copy().reshape(-1, 1)
# print("Values:",x1d)

    #K-Means Clustering
kmeans = KMeans(n_clusters=2, random_state=0, max_iter= 20).fit(x1d)
y_pred=kmeans.predict(x1d)

labels=kmeans.labels_
centroids = kmeans.cluster_centers_
    # print("Kmeans labels:",kmeans.labels_)
    # zeros=labels.count(0)
num_zeros = (labels == 0)
num_ones = (labels > 0)
    # print(num_zeros.sum(),num_ones.sum())
center= np.array(kmeans.cluster_centers_)
    
zlevel_mean=np.mean(num_zeros)
nzlevel_mean=np.mean(num_ones)
    # print("Mean values:", zlevel_mean, nzlevel_mean)
zlevel_std=np.std(num_zeros)
nzlevel_std=np.std(num_ones)
    # print("Standard Deviations:", zlevel_std, nzlevel_std)
    # print("centers:", center)
    # img=img.squeeze()
width, height = dm.shape
y_pred = y_pred.reshape(width, height)
# print('y_pred shape ', y_pred.shape)
# print("y prediction",y_pred)
a2 = centroids[labels]
# print('a2 shape ', a2.shape)
a3 = a2.reshape(width, height)
bin_img = np.where(a3== np.min(a3), 1, 0)
num_zero = (bin_img == 0).sum() #black
num_one = (bin_img != 0).sum() #white
    # print(num_zero,num_one)
bin_img=1-bin_img
    # print('a3 shape ', a3.shape)
    # print('a3 values ', a3)
dmimage=np.array(bin_img*255).astype('uint8')
    # dmimage=Image.fromarray(bin_img*255)
    # dmimage=dmimage.resize((w,h))
# print(dmimage)
plt.imshow(image)
plt.imshow(dmimage, alpha=0.6)
plt.show()

# print(binmask_img)
# binmask_img=cv2.resize(binmask_img,(w,h))
boxes=mask_to_box(dmimage,image)
print(boxes)
