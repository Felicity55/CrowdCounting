import os
import torch
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
import cv2
import torchvision.transforms as T
transform = T.Resize(size=(512,512))

class Cars(Dataset):
    """
    Vehicle counting dataset with bounding box and vehicle density map

    args:
        root_dir: the root directory of CityCam, which should be 'CityCam/'
        dataset_type: 'Train' for the training set, 'Test' for the test set
        transform: torchvision.transforms
    """
    def __init__(self, root_dir, dataset_type='Train', transform=None):
        """

        """
        self.root_dir = root_dir
        self.transform = transform
        self.sample_list = []

        train_test_split_dir = os.path.join(self.root_dir, r'E:\Dataset\CRPD_all')
        data_path = os.path.join(train_test_split_dir, r'E:\Dataset\CRPD_all\Annotation_' + dataset_type + '.txt')
        # parkway_path = os.path.join(train_test_split_dir, 'C:/Users/CVPR/source/repos/Detection/CityCam/train_test_separation/Parkway_' + dataset_type + '.txt')

        frame_dirs = []
        with open(data_path) as f:
            if dataset_type == 'Train':
                frame_dirs.extend(f.readlines()[:50])
            else:
                frame_dirs.extend(f.readlines()[:20])


        frame_dirs = [fd.strip() for fd in frame_dirs]
        for fd in frame_dirs:
            camera_dir = os.path.join(root_dir, fd.split('-')[0])
            frame_dir = os.path.join(camera_dir, fd)
            frame_names = sorted(filter(lambda fn: fn.split('.')[-1] == 'jpg', os.listdir(frame_dir)))
            self.sample_list.extend([os.path.join(frame_dir, fn) for fn in frame_names])
        

    def __len__(self):
        return len(self.sample_list)

    # def _get_vehicle_num(self, xml_path):
    #     tree = ET.parse(xml_path)
    #     root = tree.getroot()
    #     objects = root.findall('.//object')

    #     return len(objects)

    def __getitem__(self, idx):
        img = Image.open(self.sample_list[idx]).convert('RGB')
        img=img.resize((512,512))
        # img=cv2.imread(self.sample_list[idx])  #modified 13/9/2022
        np_img = np.array(img).transpose(2, 0, 1)
        # assert np_img.shape == (3, 240, 352)

        density_map = np.load(self.sample_list[idx][:-4] + '_hm'+'.npy')
        density_map = Image.fromarray(density_map)
        density_map= transform(density_map)
        density_map=np.asarray(density_map)
        # density_map = np.resize(density_map,(256,256))#21.9.22
        
        # assert density_map.shape == (240, 352)
        # assert density_map.shape == (60, 88)

        # vehicle_num = self._get_vehicle_num(self.sample_list[idx][:-4] + '.xml')
        
        sample = {'image': np_img, 'density_map': density_map}
        # sample = {'image': np_img, 'density_map': density_map, 'gt_count': vehicle_num}
        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor():
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        img = sample['image']
        img = img/255 
        density_map = sample['density_map']
        # vehicle_num = sample['gt_count']

        return {
            'image': torch.from_numpy(img).float(),
            'density_map': torch.from_numpy(density_map).float(),
            # 'gt_count': vehicle_num
        }