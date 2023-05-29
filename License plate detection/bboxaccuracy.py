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
checkpoint_path = r'C:\Users\CVPR\Soumi DI\License plate detection\checkpoint\CRPD-w015-Epochs-5000_BatchSize-6_LR-0.0001_Momentum-0.95_Gamma-0.5_Version-1\best_model.pt'
# checkpoint_path = r'C:\Users\CVPR\Soumi DI\License plate detection\checkpoint\NewdataCSRNet-w015-Epochs-7000_BatchSize-8_LR-0.0001_Momentum-0.95_Gamma-0.5_Version-1\best_model.pt'
device = torch.device('cuda:0' if use_gpu else 'cpu')


def infer(sample):
    # model = U_Net()
    
    model=CSRNet_Sig()
    # checkpoint = torch.load(checkpoint_path) #28.9.22
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda:0' if use_gpu else 'cpu')) #28.9.22// run with only cpu
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    img = sample
    img = torch.unsqueeze(img, dim=0)
    img = img.to(device) 
    
    with torch.set_grad_enabled(False):
        et_dm = model(img).detach()
        et_dm = et_dm.squeeze(0).cpu().numpy()
    
    # print(et_dm.dtype)
   
    x=et_dm.squeeze()
    x = gaussian_filter(x, sigma=10)
    
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


def mask_to_box(mask,image):
    
    h,w= mask.shape
    pred=np.zeros((h, w))
    
    # mask=Image.fromarray(mask,"L")
    largest=0
    hull = []
    border = np.zeros((h, w))
    # print(type(mask))
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
        # s=cv2.boundingRect(cn)
        ROI = image[y:y+h, x:x+w]
        
        # print("ROI:", ROI)
        # plt.imsave(r'C:\Users\CVPR\Desktop\Test images\realdata\crop\lp\053_ROI_0_lpcrop_{}.png'.format(ROI_number), ROI)
        ############# OCR detection  ####################################################################################################################
        reader = Reader(['en'])
        result = reader.readtext(ROI)
        # result = np.asarray(result, dtype = 'int')
        # print(result)
        if result != []:
            # print('Detected')
            i=100
            for detection in result:
                # print(detection[0])
                t1=np.asarray(detection[0])
                l=t1.astype(int)
                # print(type(t1))
                
                a=[]
                for i in l:
                    i=i+[x, y]
                    a.append((i[0],i[1]))
                # print("print a",a)
                a=np.asarray(a)
                # print(a[0][0],a[1][0],a[0][1],a[2][1])
                img2= cv2.polylines(copy, [a], 1, (0,255,0),2)
                pred[a[0][1]:a[2][1],a[0][0]:a[1][0]] = 1.
                i=i+10
            # t=cv2.rectangle(copy, (x,y),(x+w,y+h), (0,255,0),2)
            # plt.imshow(t)
            # plt.show()
            # plt.imshow(img2)
            # plt.show()
            # plt.imshow(pred)
            # plt.show()
            # plt.imsave(r'C:\Users\CVPR\Desktop\Test images\realdata\crop\lp\001_ROI_0_lpcrop_{}.png'.format(ROI_number), ROI)
            # plt.imsave(r'C:\Users\CVPR\Desktop\Test images\realdata\crop\lp\053_ROI_0_lpbbox_{}.png'.format(ROI_number), img2)
        # else:
        #     print('Not Detected')
        ROI_number += 1
    return pred

import os
data_trans = transforms.Compose([transforms.ToTensor()])
entries = os.listdir(r'E:\Dataset\CRPD_all\Test\Test')
precision=[]
recall=[]
f1score=[]
avgpre=[]
for file in entries:
    # Check whether file is in text format or not

    if  file.endswith(".jpg") :
        # file_path = f"{entries}\{file}"
        imgpath=os.path.join(r'E:\Dataset\CRPD_all\Test\Test',file)
        dmpath=imgpath.rsplit('.', 1)[0]+'_hm.npy'
        print(imgpath)
        image=cv2.imread(imgpath)

        image= cv2. cvtColor(image, cv2.COLOR_BGR2RGB)
        img=data_trans(image)
        hm=np.load(dmpath)
        x,y=hm.shape
        print(hm.shape)
        hm=hm.flatten()

        pred_scores=infer(img)
        x1d = pred_scores.copy().reshape(-1, 1)
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
        width, height = pred_scores.shape
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
        # plt.imshow(image)
        # plt.imshow(dmimage, alpha=0.6)
        # plt.show()
    
        # print(binmask_img)
        # binmask_img=cv2.resize(binmask_img,(w,h))
        boxes=mask_to_box(dmimage,image)
        print(boxes.shape)
        np.save(imgpath.rsplit('.', 1)[0]+'_pred.npy',boxes)
        # x1,y1=pred_scores.shape
        # pred_scores = np.pad(pred_scores, ((0, x-x1), (0, y-y1)))
        # pred_scores=pred_scores.flatten()
        # threshold =(pred_scores.max()+pred_scores.min())/1.7
        # dm = [1 if score >= threshold else 0 for score in pred_scores]
        
        x1,y1=boxes.shape
        pred_scores = np.pad(boxes, ((0, x-x1), (0, y-y1)))
        pred_scores=pred_scores.flatten()
        print(hm)
        print(pred_scores)
        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(hm, pred_scores).ravel()
        # tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
        # print(tp, fp,fn,tn)

        p=sklearn.metrics.precision_score(hm, pred_scores)
        r=sklearn.metrics.recall_score(hm, pred_scores)
        f1=sklearn.metrics.f1_score(hm, pred_scores)
        # p= (c[0][0])
        ap=sklearn.metrics.average_precision_score(hm, pred_scores)
        if p!=0.0 and r!=0.0:
            precision.append(p)
            recall.append(r)
            f1score.append(f1)
            avgpre.append(ap)
        # precision.append(p)
        # recall.append(r)
        # f1score.append(f1)
        # avgpre.append(ap)
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
# print(len(precision))

  
        # print(imgpath)
        # print(dmpath)


    


