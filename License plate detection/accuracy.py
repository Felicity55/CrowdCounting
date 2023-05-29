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
import sklearn.metrics
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
checkpoint_path = r'C:\Users\CVPR\Soumi DI\License plate detection\checkpoint\CCPD-w015-Epochs-5000_BatchSize-8_LR-1e-06_Momentum-0.95_Gamma-0.5_Version-1\best_model.pt'
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
    # print("Shape of test image", img.shape)
    
    with torch.set_grad_enabled(False):
        et_dm = model(img).detach()
        # print('etdm shape', et_dm.shape)
        # down_sample = nn.Sequential(nn.MaxUnpool2d(2), nn.MaxUnpool2d(2))
        # et_dm = down_sample(et_dm)
        
        et_dm = et_dm.squeeze(0).cpu().numpy()
        # print('etdm shape', et_dm.shape)
        # print("Estimated density map shape is:",et_dm.shape)
        # print("Estimated density map is:",et_dm)
        # print("Max value of Estimated Density Map is:",et_dm.max())
        # print("Min value of Estimated Density Map is:",et_dm.min())
        # down_gt_dm = down_gt_dm.cpu().numpy()
        # et_dm_reshape = et_dm.copy().reshape(-1, 1)
    # print(f'The integral of the estimated density map: {et_dm.sum()}')
    
    # print(et_dm.dtype)
   
    x=et_dm.squeeze()
    # x = gaussian_filter(x, sigma=10)
    
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
    bin_img = np.where(a3== np.max(a3), 1, 0)
    # num_zero = (bin_img == 0).sum() #black
    # num_one = (bin_img != 0).sum() #white
    # print(num_zero,num_one)
    # bin_img=1-bin_img
    # print('a3 shape ', a3.shape)
    # print('a3 values ', a3)
    dmimage=bin_img*255
    # dmimage=Image.fromarray(bin_img*255)
    # dmimage=dmimage.resize((w,h))
   

   
    return x
    

import os
data_trans = transforms.Compose([transforms.ToTensor()])
entries = os.listdir(r'D:/Soumi/License plate detection/LP-night/1/1/')
precision=[]
recall=[]
f1score=[]
avgpre=[]
for file in entries:
    # Check whether file is in text format or not

    if  file.endswith(".jpg") :
        # file_path = f"{entries}\{file}"
        imgpath=os.path.join('D:/Soumi/License plate detection/LP-night/1/1/',file)
        dmpath=imgpath.rsplit('.', 1)[0]+'_hm.npy'
        print(imgpath)
        image=cv2.imread(imgpath)

        image= cv2. cvtColor(image, cv2.COLOR_BGR2RGB)
        img=data_trans(image)
        hm=np.load(dmpath)
        x,y=hm.shape
        # print(hm.shape)
        hm=hm.flatten()

        pred_scores=infer(img)
        x1,y1=pred_scores.shape
        pred_scores = np.pad(pred_scores, ((0, x-x1), (0, y-y1)))
        pred_scores=pred_scores.flatten()
        threshold =(pred_scores.max()+pred_scores.min())/1.7
        dm = [1 if score >= threshold else 0 for score in pred_scores]
        # x1,y1=dm.shape
        # dm = np.pad(dm, ((0, x-x1), (0, y-y1)))
        # print(dm.shape)
        # dm=dm.flatten()

        # print("Output shape:", x,y)

        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(hm, dm).ravel()
        # tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
        # print(tp, fp,fn,tn)

        # c = np.flip(c)
        # print(c)
        # p=((c[0][0])/(c[0][0]+c[0][1]))
        # r=((c[0][0])/(c[0][0]+c[1][0]))
        # f1=((2*p*r)/(p+r))
        # acc = (c[0][0] + c[-1][-1]) / np.sum(c)
        # print(acc)

        p=sklearn.metrics.precision_score(hm, dm)
        r=sklearn.metrics.recall_score(hm, dm)
        f1=sklearn.metrics.f1_score(hm, dm)
        # p= (c[0][0])
        ap=sklearn.metrics.average_precision_score(hm, dm)
        # if p!=0.0 and r!=0.0:
        precision.append(p)
        recall.append(r)
        f1score.append(f1)
        avgpre.append(ap)
        
        print(p,r,f1,ap)
        # p=sklearn.metrics.precision_score(hm, dm)
        # r=sklearn.metrics.precision_score(hm, dm)
        # f1=sklearn.metrics.f1_score(hm, dm)
        # p= (c[0][0])
avgprecision=sum(precision)/len(precision)
avgrecall=sum(recall)/len(recall)
avgf1score=sum(f1score)/len(f1score)
avgavgpre=sum(avgpre)/len(avgpre)
print(avgprecision,avgrecall,avgf1score,avgavgpre)
print(len(precision))

  
        # print(imgpath)
        # print(dmpath)


    


# imgpath=r'D:\Soumi\License plate detection\1\1\Cars9.png'
# dmpath=r'D:\Soumi\License plate detection\1\1\Cars9_hm.npy'
# image=cv2.imread(imgpath)

# image= cv2. cvtColor(image, cv2.COLOR_BGR2RGB)
# img=data_trans(image)

# # print("Output shape:", dm)
# hm=np.load(dmpath)
# x,y=hm.shape
# print(hm.shape)
# hm=hm.flatten()

# dm=infer(img)
# x1,y1=dm.shape
# dm = np.pad(dm, ((0, x-x1), (0, y-y1)))
# print(dm.shape)
# dm=dm.flatten()

# print("Output shape:", x,y)

# c = sklearn.metrics.confusion_matrix(hm, dm)

# # c = np.flip(c)
# print(c)
# p=((c[0][0])/(c[0][0]+c[0][1]))
# r=((c[0][0])/(c[0][0]+c[1][0]))
# f1=((2*p*r)/(p+r))
# # acc = (c[0][0] + c[-1][-1]) / np.sum(c)
# # print(acc)

# ap=sklearn.metrics.average_precision_score(hm, dm)
# # p=sklearn.metrics.precision_score(hm, dm)
# # r=sklearn.metrics.precision_score(hm, dm)
# # f1=sklearn.metrics.f1_score(hm, dm)
# # p= (c[0][0])
# print(ap,p,r,f1)