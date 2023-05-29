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
import torchvision.transforms as T
transform = T.Resize(size=(1664, 1248))

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
        print(imgpath)
        # arr=np.load(imgpath)
        # # arr=arr*255
        # print(np.unique(arr))
        # print(arr.shape)
        # data = Image.fromarray(arr,'L')
        image=cv2.imread(imgpath)
        
        new_width = int(image.shape[1]/4)
        new_height = int(image.shape[0]/4)
        print(new_width,new_height)

        img_half = cv2.resize(image, (new_width, new_height))
        # data= transform(data)
        # image=np.asarray(data)
        
        
        print(img_half.shape)
        plt.imsave(imgpath,img_half)
        # dmpath=imgpath.rsplit('.', 1)[0]+'.npy'
        # np.save(dmpath,image)
        # image.show()
        # print(image.shape)