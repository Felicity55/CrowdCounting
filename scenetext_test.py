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
# checkpoint_path = r'C:\Users\CVPR\Soumi DI\License plate detection\checkpoint\NewCSRNet-w015-Epochs-5000_BatchSize-32_LR-0.01_Momentum-0.95_Gamma-0.5_Version-1\best_model.pt'
checkpoint_path = r'C:\Users\CVPR\Soumi DI\License plate detection\checkpoint\NewdataCSRNet-w015-Epochs-7000_BatchSize-8_LR-0.0001_Momentum-0.95_Gamma-0.5_Version-1\best_model.pt'
device = torch.device('cuda:0' if use_gpu else 'cpu')


def infer(sample):
    # model = U_Net()
    model=CSRNet_Sig()
    # checkpoint = torch.load(checkpoint_path) #28.9.22
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu')) #28.9.22// run with only cpu
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
        x,y,w,h = cv2.boundingRect(cn)
        ROI = image[y:y+h, x:x+w]
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
                # print(a)
                a=np.asarray(a)
                img2= cv2.polylines(copy, [a], 1, (0,255,0),2)
                i=i+10
            plt.imshow(img2)
            plt.show()
            # plt.imsave(r'C:\Users\CVPR\Desktop\Test images\realdata\crop\lp\053_ROI_0_lpbbox_{}.png'.format(ROI_number), img2)
        else:
            print('Not Detected')
        ROI_number += 1
    return 0
    




    

data_trans = transforms.Compose([transforms.ToTensor()])
# imgpath=r'C:\Users\CVPR\Desktop\Test images\img1139.jpg'
# imgpath=r'D:\Soumi\License plate detection\1\1\Cars199.png'
# imgpath=r'C:\Users\CVPR\Desktop\Test images\realdata\crop\053_ROI_0.png'
imgpath=r'C:\Users\CVPR\Desktop\Test images\realdata\img1139.jpg'
# imgpath=r'C:\Users\CVPR\source\repos\Detection\CityCam\253\253-20160421-15\000001.jpg'
# dmpath=r"C:\Users\CVPR\source\repos\Detection\CityCam\846\846-20160429-07\000001_dm.npy"
image=cv2.imread(imgpath)

image= cv2. cvtColor(image, cv2.COLOR_BGR2RGB)
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
