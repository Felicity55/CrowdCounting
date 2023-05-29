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

data_trans = transforms.Compose([transforms.ToTensor()])
entries = os.listdir('F:/datasets/CCPD2019/2/')
precision=[]
recall=[]
f1score=[]
avgpre=[]
for file in entries:
    # Check whether file is in text format or not

    if  file.endswith("_hm.npy") :
        # file_path = f"{entries}\{file}"
        imgpath=os.path.join(r'F:/datasets/CCPD2019/2/',file)
        dmpath=imgpath.rsplit('_', 1)[0]+'_pred.npy'
        # print(imgpath)
        hm=np.load(imgpath)
        predhm=np.load(dmpath)
        x,y=hm.shape
        # print(hm.shape)
        hm=hm.flatten()
        x1,y1=predhm.shape
        pred_scores = np.pad(predhm, ((0, x-x1), (0, y-y1)))
        dm=pred_scores.flatten()
        # threshold =(pred_scores.max()+pred_scores.min())/1.7
        # dm = [1 if score >= threshold else 0 for score in pred_scores]
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
        if p>=0.0 and r>=0.0:
            precision.append(p)
            recall.append(r)
            f1score.append(f1)
            avgpre.append(ap)
            print(imgpath)
        print(p,r,f1,ap)
        # p=sklearn.metrics.precision_score(hm, dm)
        # r=sklearn.metrics.precision_score(hm, dm)
        # f1=sklearn.metrics.f1_score(hm, dm)
        # p= (c[0][0])
        # print(imgpath)
avgprecision=sum(precision)/len(precision)
avgrecall=sum(recall)/len(recall)
avgf1score=sum(f1score)/len(f1score)
avgavgpre=sum(avgpre)/len(avgpre)
print(avgprecision,avgrecall,avgf1score,avgavgpre)
print(len(precision))

  

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